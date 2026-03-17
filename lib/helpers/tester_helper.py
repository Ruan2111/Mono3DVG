import os
import json
import shutil
from collections import defaultdict
from datetime import datetime

import tqdm

import torch
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.helpers.utils_helper import calc_iou
import time
import numpy as np
from utils import misc

class Tester(object):
    def __init__(self, cfg, model, dataloader, logger,loss, train_cfg=None, model_name='mono3dvg'):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'Mono3DRefer')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.model_name = model_name
        self.mono3dvg_loss = loss
        self.id2cls = {}
        if hasattr(self.dataset, 'cls2id'):
            self.id2cls = {int(cls_id): cls_name for cls_name, cls_id in self.dataset.cls2id.items()}
        self.annotation_by_sample = {}
        self.gt_labels_by_image = defaultdict(dict)
        self._build_annotation_lookup()

    def _build_annotation_lookup(self):
        anno_data = getattr(self.dataset, 'anno_data', None)
        if not anno_data:
            return

        for im_name, instance_id, ann_id, object_name, text, label_2 in anno_data:
            image_id = int(im_name)
            sample_id = self._make_sample_id(image_id, instance_id, ann_id)
            label_values = self._normalize_gt_label(label_2)
            self.annotation_by_sample[sample_id] = {
                'sample_id': sample_id,
                'image_id': image_id,
                'instance_id': int(instance_id),
                'anno_id': int(ann_id),
                'object_name': str(object_name),
                'text': text,
                'label_2': label_values,
            }
            self.gt_labels_by_image[image_id][int(instance_id)] = label_values

    @staticmethod
    def _make_sample_id(image_id, instance_id, ann_id):
        return f"{int(image_id)}_{int(instance_id)}_{int(ann_id)}"

    @staticmethod
    def _normalize_gt_label(label_2):
        return [str(label_2[0])] + [float(value) for value in label_2[1:]]

    @staticmethod
    def _reset_dir(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _symlink_or_copy(src_path, dst_path):
        if os.path.lexists(dst_path):
            os.remove(dst_path)
        try:
            os.symlink(os.path.abspath(src_path), dst_path)
        except OSError:
            shutil.copy2(src_path, dst_path)

    @staticmethod
    def _metric_summary(iou_values):
        if not iou_values:
            return {'acc25': 0.0, 'acc5': 0.0, 'miou': 0.0}
        iou_array = np.array(iou_values, dtype=np.float32)
        return {
            'acc25': float((iou_array > 0.25).mean() * 100.0),
            'acc5': float((iou_array > 0.5).mean() * 100.0),
            'miou': float(iou_array.mean() * 100.0),
        }

    def _get_class_name(self, cls_id):
        return self.id2cls.get(int(cls_id), str(int(cls_id)))

    @staticmethod
    def _to_kitti_class_name(class_name):
        mapping = {
            'pedestrian': 'Pedestrian',
            'car': 'Car',
            'cyclist': 'Cyclist',
            'van': 'Van',
            'truck': 'Truck',
            'tram': 'Tram',
            'bus': 'Bus',
            'person_sitting': 'Person_sitting',
            'motorcyclist': 'Motorcyclist',
        }
        return mapping.get(class_name.lower(), class_name)

    def _format_kitti_line(self, label_values):
        class_name = str(label_values[0])
        truncation = float(label_values[1])
        occlusion = int(round(float(label_values[2])))
        numeric_values = [float(value) for value in label_values[3:]]
        numeric_str = " ".join(f"{value:.6f}" for value in numeric_values)
        return f"{class_name} {truncation:.2f} {occlusion:d} {numeric_str}".rstrip()

    def _prediction_to_kitti_values(self, record):
        sample_meta = self.annotation_by_sample.get(record['sample_id'])
        truncation = float(sample_meta['label_2'][1]) if sample_meta else 0.0
        occlusion = int(round(float(sample_meta['label_2'][2]))) if sample_meta else 0
        pred = record['prediction']
        return [
            self._to_kitti_class_name(pred['class_name']),
            truncation,
            occlusion,
            float(pred['alpha']),
            *[float(value) for value in pred['bbox_2d_xyxy']],
            *[float(value) for value in pred['dimensions_hwl']],
            *[float(value) for value in pred['location_xyz_bottom']],
            float(pred['rotation_y']),
            float(pred['score']),
        ]

    def _get_query_gt_line(self, sample_id):
        sample_meta = self.annotation_by_sample.get(sample_id)
        if not sample_meta:
            return None
        return self._format_kitti_line(sample_meta['label_2'])

    def _get_image_gt_lines(self, image_id):
        gt_by_instance = self.gt_labels_by_image.get(int(image_id), {})
        return [self._format_kitti_line(label_2) for _, label_2 in sorted(gt_by_instance.items())]

    @staticmethod
    def _select_best_records_per_instance(records):
        best_records = {}
        for record in records:
            key = (int(record['image_id']), int(record['instance_id']))
            current_record = best_records.get(key)
            if current_record is None:
                best_records[key] = record
                continue

            current_score = float(current_record['prediction']['score'])
            candidate_score = float(record['prediction']['score'])
            if candidate_score > current_score:
                best_records[key] = record

        records_by_image = defaultdict(list)
        for (_, _), record in best_records.items():
            image_key = f"{int(record['image_id']):06d}"
            records_by_image[image_key].append(record)

        for image_key in records_by_image:
            records_by_image[image_key].sort(key=lambda item: (int(item['instance_id']), int(item['anno_id'])))

        return records_by_image

    def _build_subset_membership(self, instance_id, split_lookup):
        instance_id = int(instance_id)
        uniqueness = "Unique" if instance_id in split_lookup['Unique'] else "Multiple"

        if instance_id in split_lookup['Near']:
            distance = "Near"
        elif instance_id in split_lookup['Medium']:
            distance = "Medium"
        else:
            distance = "Far"

        if instance_id in split_lookup['Easy']:
            difficulty = "Easy"
        elif instance_id in split_lookup['Moderate']:
            difficulty = "Moderate"
        else:
            difficulty = "Hard"

        return {
            'uniqueness': uniqueness,
            'distance': distance,
            'difficulty': difficulty,
        }

    def _build_visualization_record(self, sample_id, image_id, instance_id, ann_id, text, img_size, pred, gt_box3d, iou_3d, split_lookup):
        cls_id = int(pred[0])
        class_name = self._get_class_name(cls_id)
        pred_bbox = [float(value) for value in pred[2:6]]
        pred_dims = [float(value) for value in pred[6:9]]
        pred_loc = [float(value) for value in pred[9:12]]
        pred_ry = float(pred[12])
        pred_score = float(pred[13])

        return {
            'sample_id': sample_id,
            'image_id': int(image_id),
            'image_path': os.path.join('.', self.dataset.root_dir, 'images', f'{int(image_id):06d}.png'),
            'instance_id': int(instance_id),
            'anno_id': int(ann_id),
            'text': text,
            'img_size': [int(img_size[0]), int(img_size[1])],
            'subsets': self._build_subset_membership(instance_id, split_lookup),
            'prediction': {
                'class_id': cls_id,
                'class_name': class_name,
                'alpha': float(pred[1]),
                'bbox_2d_xyxy': pred_bbox,
                'dimensions_hwl': pred_dims,
                'location_xyz_bottom': pred_loc,
                'rotation_y': pred_ry,
                'score': pred_score,
            },
            'ground_truth': {
                'box_3d_hwlxyz_bottom': [float(value) for value in gt_box3d],
            },
            'iou_3d': float(iou_3d),
        }

    def _export_kitti_predictions(self, records):
        pred_dir = os.path.join(self.output_dir, 'kitti_pred')
        self._reset_dir(pred_dir)

        best_records_by_image = self._select_best_records_per_instance(records)
        metadata_by_image = {}
        for image_key, image_records in best_records_by_image.items():
            metadata_by_image[image_key] = []
            for record in image_records:
                metadata_by_image[image_key].append({
                    'sample_id': record['sample_id'],
                    'instance_id': record['instance_id'],
                    'anno_id': record['anno_id'],
                    'text': record['text'],
                    'score': record['prediction']['score'],
                    'iou_3d': record['iou_3d'],
                })

        for image_key, image_records in best_records_by_image.items():
            pred_lines = [self._format_kitti_line(self._prediction_to_kitti_values(record)) for record in image_records]
            pred_path = os.path.join(pred_dir, f'{image_key}.txt')
            with open(pred_path, 'w') as pred_file:
                pred_file.write("\n".join(pred_lines))
                pred_file.write("\n")

        meta_path = os.path.join(self.output_dir, 'kitti_pred_meta.json')
        with open(meta_path, 'w') as meta_file:
            json.dump({
                'selection_policy': 'max_score_per_image_instance',
                'records': metadata_by_image,
            }, meta_file, indent=2)

        self.logger.info(f"Saved KITTI-format predictions to {pred_dir}")
        self.logger.info(f"Saved KITTI prediction metadata to {meta_path}")

    def _export_visualization_json(self, checkpoint_name, records, summary):
        vis_json_path = os.path.join(self.output_dir, f'{checkpoint_name}_visualization.json')
        payload = {
            'model_name': self.model_name,
            'checkpoint': self.cfg['pretrain_model'],
            'generated_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'num_samples': len(records),
            'summary': summary,
            'records': records,
        }
        with open(vis_json_path, 'w') as vis_file:
            json.dump(payload, vis_file, indent=2)
        self.logger.info(f"Saved visualization results to {vis_json_path}")

    def _export_kitti_vis_workspace(self, records):
        vis_root = os.path.join(self.output_dir, 'kitti_vis')
        training_dir = os.path.join(vis_root, 'training')
        image_dir = os.path.join(training_dir, 'image_2')
        calib_dir = os.path.join(training_dir, 'calib')
        label_dir = os.path.join(training_dir, 'label_2')
        velo_dir = os.path.join(training_dir, 'velodyne')
        pred_dir = os.path.join(training_dir, 'pred')
        by_query_root = os.path.join(vis_root, 'by_query')

        self._reset_dir(vis_root)
        os.makedirs(training_dir, exist_ok=True)
        for path in [image_dir, calib_dir, label_dir, velo_dir, pred_dir, by_query_root]:
            os.makedirs(path, exist_ok=True)

        best_records_by_image = self._select_best_records_per_instance(records)
        used_image_ids = sorted({int(record['image_id']) for record in records})
        for image_id in used_image_ids:
            image_key = f"{image_id:06d}"
            src_image = os.path.join(self.dataset.root_dir, 'images', f'{image_key}.png')
            src_calib = os.path.join(self.dataset.root_dir, 'calib', f'{image_key}.txt')
            self._symlink_or_copy(src_image, os.path.join(image_dir, f'{image_key}.png'))
            self._symlink_or_copy(src_calib, os.path.join(calib_dir, f'{image_key}.txt'))

            with open(os.path.join(label_dir, f'{image_key}.txt'), 'w') as label_file:
                gt_lines = self._get_image_gt_lines(image_id)
                if gt_lines:
                    label_file.write("\n".join(gt_lines))
                    label_file.write("\n")

            pred_lines = [self._format_kitti_line(self._prediction_to_kitti_values(record))
                          for record in best_records_by_image.get(image_key, [])]
            with open(os.path.join(pred_dir, f'{image_key}.txt'), 'w') as pred_file:
                if pred_lines:
                    pred_file.write("\n".join(pred_lines))
                    pred_file.write("\n")

            open(os.path.join(velo_dir, f'{image_key}.bin'), 'wb').close()

        query_manifest = []
        for record in records:
            sample_id = record['sample_id']
            image_key = f"{int(record['image_id']):06d}"
            sample_root = os.path.join(by_query_root, sample_id)
            sample_training_dir = os.path.join(sample_root, 'training')
            sample_image_dir = os.path.join(sample_training_dir, 'image_2')
            sample_calib_dir = os.path.join(sample_training_dir, 'calib')
            sample_label_dir = os.path.join(sample_training_dir, 'label_2')
            sample_velo_dir = os.path.join(sample_training_dir, 'velodyne')
            sample_pred_dir = os.path.join(sample_training_dir, 'pred')
            for path in [sample_image_dir, sample_calib_dir, sample_label_dir, sample_velo_dir, sample_pred_dir]:
                os.makedirs(path, exist_ok=True)

            src_image = os.path.join(self.dataset.root_dir, 'images', f'{image_key}.png')
            src_calib = os.path.join(self.dataset.root_dir, 'calib', f'{image_key}.txt')
            self._symlink_or_copy(src_image, os.path.join(sample_image_dir, f'{image_key}.png'))
            self._symlink_or_copy(src_calib, os.path.join(sample_calib_dir, f'{image_key}.txt'))

            gt_line = self._get_query_gt_line(sample_id)
            with open(os.path.join(sample_label_dir, f'{image_key}.txt'), 'w') as label_file:
                if gt_line:
                    label_file.write(gt_line)
                    label_file.write("\n")

            with open(os.path.join(sample_pred_dir, f'{image_key}.txt'), 'w') as pred_file:
                pred_file.write(self._format_kitti_line(self._prediction_to_kitti_values(record)))
                pred_file.write("\n")

            open(os.path.join(sample_velo_dir, f'{image_key}.bin'), 'wb').close()

            query_manifest.append({
                'sample_id': sample_id,
                'image_id': int(record['image_id']),
                'instance_id': int(record['instance_id']),
                'anno_id': int(record['anno_id']),
                'text': record['text'],
                'score': float(record['prediction']['score']),
                'iou_3d': float(record['iou_3d']),
                'workspace_dir': sample_root,
                'pred_file': os.path.join(sample_pred_dir, f'{image_key}.txt'),
                'label_file': os.path.join(sample_label_dir, f'{image_key}.txt'),
            })

        manifest_path = os.path.join(vis_root, 'query_manifest.json')
        with open(manifest_path, 'w') as manifest_file:
            json.dump(query_manifest, manifest_file, indent=2)

        self.logger.info(f"Saved KITTI viewer workspace to {vis_root}")
        self.logger.info(f"Saved query-wise predictions to {by_query_root}")
        self.logger.info(f"Saved query manifest to {manifest_path}")

    def test(self):
        # test a checkpoint
        checkpoint_path = os.path.join(self.output_dir, self.cfg['pretrain_model'])
        assert os.path.exists(checkpoint_path)
        load_checkpoint(model=self.model,
                        optimizer=None,
                        filename=checkpoint_path,
                        map_location=self.device,
                        logger=self.logger)
        self.model.to(self.device)
        self.inference()

    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        with open("Mono3DRefer/test_instanceID_split.json", "r") as file:
            test_instanceID_spilt = {key: set(values) for key, values in json.load(file).items()}
        results = {}
        iou_3dbox_test = {"Unique": [], "Multiple": [], "Overall": [], "Near": [], "Medium": [], "Far": [],
                          "Easy": [], "Moderate": [], "Hard": []}
        vis_records = []
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)
            gt_3dboxes = info['gt_3dbox']
            gt_3dboxes = [[float(gt_3dboxes[j][i].detach().cpu().numpy()) for j in range(len(gt_3dboxes)) ] for i in range(len(gt_3dboxes[0]))]

            captions = targets["text"]
            im_name = targets['image_id']
            instanceID = targets['instance_id']
            ann_id = targets['anno_id']
            batch_size = inputs.shape[0]

            start_time = time.time()
            outputs = self.model(inputs, calibs,  img_sizes, captions, im_name, instanceID, ann_id)
            end_time = time.time()
            model_infer_time += end_time - start_time

            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg['topk'])

            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items() if key not in ['gt_3dbox']}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets, pred_3dboxes = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,)

            for i in range(batch_size):
                image_id = int(info['img_id'][i])
                instanceID = int(info['instance_id'][i])
                anno_id = int(info['anno_id'][i])
                sample_id = self._make_sample_id(image_id, instanceID, anno_id)
                pred = dets[sample_id]

                pre_box3D = np.array(pred_3dboxes[i])
                pre_box3D = np.array([pre_box3D[3], pre_box3D[4], pre_box3D[5], pre_box3D[0],pre_box3D[1], pre_box3D[2]], dtype=np.float32)
                gt_box3D = np.array(gt_3dboxes[i])
                gt_box3D = np.array([gt_box3D[3], gt_box3D[4], gt_box3D[5], gt_box3D[0], gt_box3D[1], gt_box3D[2]],dtype=np.float32)

                gt_box3D[1] -= gt_box3D[3] / 2    # real 3D center in 3D space
                pre_box3D[1] -= pre_box3D[3] / 2  # real 3D center in 3D space
                IoU = calc_iou(pre_box3D, gt_box3D)

                iou_3dbox_test["Overall"].append(IoU)
                if instanceID in test_instanceID_spilt['Unique']:
                    iou_3dbox_test["Unique"].append(IoU)
                else:
                    iou_3dbox_test["Multiple"].append(IoU)

                if instanceID in test_instanceID_spilt['Near']:
                    iou_3dbox_test["Near"].append(IoU)
                elif instanceID in test_instanceID_spilt['Medium']:
                    iou_3dbox_test["Medium"].append(IoU)
                elif instanceID in test_instanceID_spilt['Far']:
                    iou_3dbox_test["Far"].append(IoU)

                if instanceID in test_instanceID_spilt['Easy']:
                    iou_3dbox_test["Easy"].append(IoU)
                elif instanceID in test_instanceID_spilt['Moderate']:
                    iou_3dbox_test["Moderate"].append(IoU)
                elif instanceID in test_instanceID_spilt['Hard']:
                    iou_3dbox_test["Hard"].append(IoU)

                vis_records.append(self._build_visualization_record(
                    sample_id=sample_id,
                    image_id=image_id,
                    instance_id=instanceID,
                    ann_id=anno_id,
                    text=captions[i],
                    img_size=info['img_size'][i],
                    pred=pred,
                    gt_box3d=gt_3dboxes[i],
                    iou_3d=IoU,
                    split_lookup=test_instanceID_spilt,
                ))

            results.update(dets)
            progress_bar.update()

            if batch_idx % 30 == 0:
                acc5 = np.sum(np.array((np.array(iou_3dbox_test["Overall"]) > 0.5), dtype=float)) / len(iou_3dbox_test["Overall"])
                acc25 = np.sum(np.array((np.array(iou_3dbox_test["Overall"]) > 0.25), dtype=float)) / len(iou_3dbox_test["Overall"])
                miou = sum(iou_3dbox_test["Overall"]) / len(iou_3dbox_test["Overall"])

                print_str ='Epoch: [{}/{}]\t' \
                           'Accu25 {acc25:.2f}%\t' \
                           'Accu5 {acc5:.2f}%\t' \
                           'Mean_iou {miou:.2f}%\t' \
                    .format(
                    batch_idx, len(self.dataloader), \
                    acc25=acc25 * 100, acc5=acc5 * 100, miou=miou * 100
                )
                # print(print_str)
                self.logger.info(print_str)

        num_queries = len(vis_records)
        mean_batch_time = model_infer_time / max(len(self.dataloader), 1)
        mean_query_time = model_infer_time / max(num_queries, 1)
        print("inference on {} queries across {} batches, batch time {}, query time {}".format(
            num_queries, len(self.dataloader), mean_batch_time, mean_query_time))
        progress_bar.close()

        # save the result for evaluation.
        self.logger.info('==> Mono3DVG Evaluation ...')

        summary = {}
        for split_name, iou_3dbox in iou_3dbox_test.items():
            print("------------" + split_name + "------------")
            metrics = self._metric_summary(iou_3dbox)
            print_str = 'Accu25 {acc25:.2f}%\t' \
                        'Accu5 {acc5:.2f}%\t' \
                        'Mean_iou {miou:.2f}%\t' \
                .format(
                acc25=metrics['acc25'], acc5=metrics['acc5'], miou=metrics['miou']
            )
            print(print_str)
            self.logger.info(f"{split_name}: {print_str}")
            summary[split_name] = metrics

        checkpoint_name = os.path.splitext(os.path.basename(self.cfg['pretrain_model']))[0]
        self._export_visualization_json(checkpoint_name, vis_records, summary)
        self._export_kitti_predictions(vis_records)
        self._export_kitti_vis_workspace(vis_records)


    def evaluate(self, epoch):
        torch.set_grad_enabled(False)
        self.model.eval()

        iou_3dbox = []
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)
            gt_3dboxes = info['gt_3dbox']
            gt_3dboxes = [[float(gt_3dboxes[j][i].detach().cpu().numpy()) for j in range(len(gt_3dboxes)) ] for i in range(len(gt_3dboxes[0]))]

            captions = targets["text"]
            im_name = targets['image_id']
            instanceID = targets['instance_id']
            ann_id = targets['anno_id']
            batch_size = inputs.shape[0]

            for key in targets.keys():
                if key not in ['image_id', 'text']:
                    targets[key] = targets[key].to(self.device)
            targets = self.prepare_targets(targets, batch_size)

            outputs = self.model(inputs, calibs,  img_sizes, captions, im_name, instanceID, ann_id)

            # compute Loss
            mono3dvg_losses_dict = self.mono3dvg_loss(outputs, targets)
            weight_dict = self.mono3dvg_loss.weight_dict
            mono3dvg_losses_dict = misc.reduce_dict(mono3dvg_losses_dict)
            mono3dvg_losses_dict_log = {}
            mono3dvg_losses_log = 0
            for k in mono3dvg_losses_dict.keys():
                if k in weight_dict:
                    mono3dvg_losses_dict_log[k] = (mono3dvg_losses_dict[k] * weight_dict[k]).item()
                    mono3dvg_losses_log += mono3dvg_losses_dict_log[k]
            mono3dvg_losses_dict_log["loss_mono3dvg"] = mono3dvg_losses_log

            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg['topk'])
            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items() if key not in ['gt_3dbox']}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets, pred_3dboxes = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,)

            for i in range(batch_size):
                pre_box3D = np.array(pred_3dboxes[i])
                pre_box3D = np.array([pre_box3D[3], pre_box3D[4], pre_box3D[5], pre_box3D[0], pre_box3D[1], pre_box3D[2]],dtype=np.float32)
                gt_box3D = np.array(gt_3dboxes[i])
                gt_box3D = np.array([gt_box3D[3], gt_box3D[4], gt_box3D[5], gt_box3D[0], gt_box3D[1], gt_box3D[2]],dtype=np.float32)

                gt_box3D[1] -= gt_box3D[3] / 2    # real 3D center in 3D space
                pre_box3D[1] -= pre_box3D[3] / 2  # real 3D center in 3D space
                IoU = calc_iou(pre_box3D, gt_box3D)

                iou_3dbox.append(IoU)

            if batch_idx % 30 == 0:
                acc5 = np.sum(np.array((np.array(iou_3dbox) > 0.5), dtype=float)) / len(iou_3dbox)
                acc25 = np.sum(np.array((np.array(iou_3dbox) > 0.25), dtype=float)) / len(iou_3dbox)
                miou = sum(iou_3dbox) / len(iou_3dbox)

                print_str ='Evaluation: [{}][{}/{}]\t' \
                           'Loss_mono3dvg: {:.2f}\t' \
                           'Accu25 {:.2f}%\t' \
                           'Accu5 {:.2f}%\t' \
                           'Mean_iou {:.2f}%\t' \
                    .format(
                    epoch, batch_idx, len(self.dataloader), \
                    mono3dvg_losses_dict_log["loss_mono3dvg"], \
                    acc25 * 100, acc5 * 100,\
                    miou * 100
                )
                # print(print_str)
                self.logger.info(print_str)

        acc5 = np.sum(np.array((np.array(iou_3dbox) > 0.5), dtype=float)) / len(iou_3dbox)
        acc25 = np.sum(np.array((np.array(iou_3dbox) > 0.25), dtype=float)) / len(iou_3dbox)
        miou = sum(iou_3dbox) / len(iou_3dbox)
        print_str = 'Loss_mono3dvg: {:.2f}\t' \
                    'Accu25 {:.2f}%\t' \
                    'Accu5 {:.2f}%\t' \
                    'Mean_iou {:.2f}%\t' \
            .format(
            mono3dvg_losses_dict_log["loss_mono3dvg"],
            acc25 * 100, acc5 * 100,
            miou * 100
        )
        # print("Final Evaluation Result: ",print_str)
        self.logger.info("Final Evaluation Result: "+ print_str)
        return acc25* 100, acc5* 100, mono3dvg_losses_dict_log["loss_mono3dvg"]

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
            targets_list.append(target_dict)
        return targets_list
