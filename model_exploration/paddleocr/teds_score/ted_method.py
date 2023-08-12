import os
import cv2
import pickle
from tqdm import tqdm
from teds_score.table.table_metric.table_metric import TEDS


from table.predict_table import TableSystem
from utility import init_args

class EvaluatorTEDSCUstom:
    def __init__(self, output_folder, det_model_dir, rec_model_dir, table_model_dir,
                 rec_char_dict_path, table_char_dict_path, det_limit_side_len,
                 det_limit_type):
        self.output_folder = output_folder
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.table_model_dir = table_model_dir
        self.rec_char_dict_path = rec_char_dict_path
        self.table_char_dict_path = table_char_dict_path
        self.det_limit_side_len = det_limit_side_len
        self.det_limit_type = det_limit_type

    def load_result(self, path):
        data = {}
        if os.path.exists(path):
            data = pickle.load(open(path, 'rb'))
        return data

    def save_result(self, path, data):
        old_data = self.load_result(path)
        old_data.update(data)
        with open(path, 'wb') as f:
            pickle.dump(old_data, f)

    def evaluate_table(self, gt_html_dict, img_root):
        args = init_args()
        args.output = self.output_folder
        args.det_model_dir = self.det_model_dir
        args.rec_model_dir = self.rec_model_dir
        args.table_model_dir = self.table_model_dir
        args.rec_char_dict_path = self.rec_char_dict_path
        args.table_char_dict_path = self.table_char_dict_path
        args.det_limit_side_len = self.det_limit_side_len
        args.det_limit_type = self.det_limit_type
        
        # init TableSystem
        text_sys = TableSystem(args)

        ocr_result = self.load_result(os.path.join(args.output, 'ocr.pickle'))
        structure_result = self.load_result(os.path.join(args.output, 'structure.pickle'))

        pred_htmls = []
        gt_htmls = []
        for img_name, gt_html in tqdm(gt_html_dict.items()):
            img = cv2.imread(os.path.join(img_root, img_name))
            # run ocr and save result
            if img_name not in ocr_result:
                dt_boxes, rec_res, _, _ = text_sys._ocr(img)
                ocr_result[img_name] = [dt_boxes, rec_res]
                self.save_result(os.path.join(args.output, 'ocr.pickle'), ocr_result)
            # run structure and save result
            if img_name not in structure_result:
                structure_res, _ = text_sys._structure(img)
                structure_result[img_name] = structure_res
                self.save_result(os.path.join(args.output, 'structure.pickle'), structure_result)
            dt_boxes, rec_res = ocr_result[img_name]
            structure_res = structure_result[img_name]
            # match ocr and structure
            pred_html = text_sys.match(structure_res, dt_boxes, rec_res)
            pred_htmls.append(pred_html)
            gt_htmls.append(gt_html)

        # compute teds
        teds = TEDS(n_jobs=16)
        scores = teds.batch_evaluate_html(gt_htmls, pred_htmls)
        return scores

# Example usage
if __name__ == '__main__':
    output_folder = "path/to/output_folder"
    det_model_dir = "path/to/det_model_dir"
    rec_model_dir = "path/to/rec_model_dir"
    table_model_dir = "path/to/table_model_dir"
    rec_char_dict_path = "../ppocr/utils/dict/table_dict.txt"
    table_char_dict_path = "../ppocr/utils/dict/table_structure_dict.txt"
    det_limit_side_len = 736
    det_limit_type = "min"
    
    evaluator = TableEvaluator(output_folder, det_model_dir, rec_model_dir, table_model_dir,
                               rec_char_dict_path, table_char_dict_path, det_limit_side_len,
                               det_limit_type)
    
    gt_html_dict = evaluator.load_txt("path/to/gt.txt")
    img_root = "docs/table/"
    scores = evaluator.evaluate_table(gt_html_dict, img_root)
    print(scores)
