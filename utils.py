import pandas as pd
import json
import config
import os

class data_loader:
    def __init__(self, questions_path, answers_path):
        self.data_qns = json.load(open(questions_path,"r"))
        self.data_ans = json.load(open(answers_path,"r"))
        self.df_q = data_qns['questions']
        self.df_a = data_ans['annotations']
        
    def add_question(self, image_id, qn, qn_id=None):
        # df: ['question','image_id','question_id']
        if not qn_id:
            qn_id = int(self.df_q['question_id'].max()) +1
            
        # Find index in the dataframe of image where questions for this image are located.
        image_index = df_q.index[self.df_q['image_id'] == image_id].tolist()[0]
        df_ = pd.DataFrame({'image_id':image_id,'question':qn,'question_id':qn_id},index=[0])

        # squeeze the new question in there.
        self.df_q = concat([self.df_q.ix[:image_index-1], 
                            df_, self.df_q.ix[image_index:]]).reset_index(drop=True)
        return qn_id
        
    def add_answer(answers, qid, img_id, multiple_choice = '',  atype = '', qn_type = ''):
        # df: question_type, answers, multiple_choice_answer, question_id, answer_type, image_id

        image_index = self.df_a.index[self.df_a['image_id'] == img_id].tolist()[0]
        df_ = pd.DataFrame({'question_type':qn_type,'answers':answers,
                            'multiple_choice_answer':multiple_choice, 'question_id': qid,
                           'answer_type':atype,'image_id':img_id},index=[0])
        self.df_a = concat([self.df_a.ix[:image_index-1], df_, 
                            self.df_q.ix[image_index:]]).reset_index(drop=True)
        
        
    def get_questions_from_image(self,image_id):
        return self.df_q.question[df_q['image_id']==image_id].tolist()
    
    def get_answers_from_image(self, qid):
        return self.df_a.answers[df_a['question_id']==qid].tolist()
        
        
    def get_image_ids(self):
        return self.df_q['image_id'].tolist()
        
    def get_question_ids():
        return self.df_q['question_id'].tolist()
    
    def get_questions(self):
        return self.df_q['question'].tolist()
        
        
    def dump_ans_train_json(filename):
        stringobj = self.df_a.to_json(orient="records")
        dict_json = {}
        dict_json['license'] = {'name': 'Creative Commons Attribution 4.0 International License', 'url': 'http://creativecommons.org/licenses/by/4.0/'}
        dict_json['annotations'] = stringobj
        dict_json['data_subtype'] = 'train2014'
        dict_json['info'] = {'version': '2.0', 'description': 'This is v2.0 of the VQA dataset.', 'contributor': 'VQA Team', 'url': 'http://visualqa.org', 'date_created': '2017-04-26 17:07:13', 'year': 2017}
        dict_json['data_type'] = 'mscoco'       
        with open(filename, 'w') as f:
            f.write(dict_json)
            
            
            
    def dump_qns_train_json(filename):
        stringobj = self.df_q.to_json(orient="records")
        dict_json = {}
        dict_json['license'] = {'name': 'Creative Commons Attribution 4.0 International License', 'url': 'http://creativecommons.org/licenses/by/4.0/'}
        dict_json['questions'] = stringobj
        dict_json['task_type'] = 'Open-Ended'
        dict_json['data_subtype'] = 'train2014'
        dict_json['info'] = {'year': 2017, 'description': 'This is v2.0 of the VQA dataset.', 'contributor': 'VQA Team', 'version': '2.0', 'date_created': '2017-04-26 17:07:13', 'url': 'http://visualqa.org'}
        dict_json['data_type'] = 'mscoco'
        with open(filename, 'w') as f:
            f.write(dict_json)

def helper_ans_string(answers,answers_conf = None):
    
    list_of_dicts = []
    # Assume dominant answer is the first answer.
    
    if len(answers) == 0:
        raise 'No answers given. Pl check.'
        
    if answers_conf:
        if len(answers)!=len(answers_conf): raise 'Length of answers and answers_conf incompatible.'
    
    for i, ans in enumerate(answers):
        dict_ans = {}
        dict_ans['answer_id'] = i
        dict_ans['answer'] = ans
        if answers_conf:
            dict_ans['answer_confidence'] = answers_conf[i]
        else:
            dict_ans['answer_confidence'] = 'yes'
        list_of_dicts.append(dict_ans)
            
    if len(answers)>10:
        raise "Number of answers > 10. Please truncate."
    elif len(list_of_dicts)<10:
        for i in range(len(answers),11):
            dict_ans = {}
            dict_ans['answer_id'] = i
            dict_ans['answer'] = answers[0]
            if answers_conf:
                dict_ans['answer_confidence'] = answers_conf[0]
            else:
                dict_ans['answer_confidence'] = 'yes'
            list_of_dicts.append(dict_ans)
            
            
    return list_of_dicts

# TODO: add class functionality
# answers, qid, img_id, multiple_choice = '',  atype = '', qn_type = ''):
DataLoader = dataloader(args.question_path, args.answer_path)

 def add_to_dataset(data):
        for k,v in data:
            image_id = k
            for (ques, ans) in v:
                qid = DataLoader.add_question(image_id,ques)
                DataLoader.add_answer(ans,qid, image_id)
                
        


    

