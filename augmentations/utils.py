import pandas as pd
import json
import config
import os

class data_loader:
    """ Loads training and validation json files and adds augmented questions to them. """
    def __init__(self, questions_path, answers_path):
        self.qns = json.load(open(questions_path,"r"))
        self.ans = json.load(open(answers_path,"r"))
        
        self.df_q = pd.DataFrame(self.qns['questions'])
        self.df_a = pd.DataFrame(self.ans['annotations'])
        self.max_qid = max(map(lambda x: x['question_id'], self.qns['questions']))
        
    def add_question(self, image_id, qn, qn_id=None):
        """ Appends a question to existing pandas dataframe self.data_qns.
            Can be dumped into json file by calling dump function.
            dataframe of questions contains columns: ['question','image_id','question_id']
            Returns Question ID.
        """
        
        if not qn_id:
            qn_id = self.max_qid + 1
            
        # Find index in the dataframe of image where questions for this image are located.
#         image_index = self.df_q.index[self.df_q['image_id'] == image_id].tolist()[0]
#         df_ = pd.DataFrame({'image_id':image_id,'question':qn,'question_id':qn_id},index=[0])

        # squeeze the new question in there.
#         self.df_q = pd.concat([self.df_q.ix[:image_index-1], 
#                             df_, self.df_q.ix[image_index:]]).reset_index(drop=True)
        self.qns['questions'].append({'image_id':image_id,'question':qn,'question_id':qn_id})
        self.max_qid = max(self.max_qid, qn_id)
        return qn_id
        
    def add_answer(self, answers, qid, img_id, multiple_choice = '',  atype = '', qn_type = ''):
        """ Appends an answer to existing data frame self.data_ans.
            Can be dumped into json file by calling dump function.
            dataframe of answers contains columns: [question_type, answers, multiple_choice_answer,
            question_id, answer_type, image_id]
        """

        image_index = self.df_a.index[self.df_a['image_id'] == img_id].tolist()[0]
        self.ans['annotations'].append({'question_type':qn_type,'answers':answers,
                                         'multiple_choice_answer':multiple_choice, 'question_id': qid,
                                         'answer_type':atype,'image_id':img_id})
        #df_ = pd.DataFrame({'question_type':qn_type,'answers':answers,
        #                    'multiple_choice_answer':multiple_choice, 'question_id': qid,
        #                   'answer_type':atype,'image_id':img_id},index=[0])
        #self.df_a = pd.concat([self.df_a.ix[:image_index-1], df_, 
        #                    self.df_a.ix[image_index:]]).reset_index(drop=True)
        
        
        
    def get_questions_from_image(self,image_id):
        """ Gets list of Questions for a particular Image given the Image_ID.
        """
        return self.df_q.question[self.df_q['image_id']==image_id].tolist()
    
    def get_answers_from_image(self, qid):
        """ Gets list of Answers for a particular Question given the Question_ID.
        """        
        return self.df_a.answers[self.df_a['question_id']==qid].tolist()
        
        
    def get_image_ids(self):
        """ Gets list of Image Ids. 
        """        
        return self.df_q['image_id'].tolist()
        
    def get_question_ids(self):
        """ Gets list of Question Ids. 
        """         
        return self.df_q['question_id'].tolist()
    
    def get_questions(self):
        """ Gets list of Questions. 
        """         
        return self.df_q['question'].tolist()
    
        
    def dump_ans_train_json(self, filename):
        """ Dumps the current pandas dataframe df_a with additional data to a json file.
            Matches the original formats.
            List of Keys of dumped Json: license, annotations, data_subtype, info, data_type.
        """         
#         stringobj = self.df_a.to_json(orient="records")
#         dict_json = {}
#         dict_json['license'] = {'name': 'Creative Commons Attribution 4.0 International License', 'url': 'http://creativecommons.org/licenses/by/4.0/'}
#         dict_json['annotations'] = json.loads(stringobj)
#         dict_json['data_subtype'] = 'train2014'
#         dict_json['info'] = {'version': '2.0', 'description': 'This is v2.0 of the VQA dataset.', 'contributor': 'VQA Team', 'url': 'http://visualqa.org', 'date_created': '2017-04-26 17:07:13', 'year': 2017}
#         dict_json['data_type'] = 'mscoco'       
        with open(filename, 'w') as f:
            json.dump(self.ans, f)
            
            
            
    def dump_qns_train_json(self, filename):
        """ Dumps the current pandas dataframe df_q with additional data to a json file.
            Matches the original formats.
            List of Keys of dumped Json: license, questions, task_type, data_subtype, info, data_type.
        """           
#         stringobj = self.df_q.to_json(orient="records")
#         dict_json = {}
#         dict_json['license'] = {'name': 'Creative Commons Attribution 4.0 International License', 'url': 'http://creativecommons.org/licenses/by/4.0/'}
#         dict_json['questions'] = json.loads(stringobj)
#         dict_json['task_type'] = 'Open-Ended'
#         dict_json['data_subtype'] = 'train2014'
#         dict_json['info'] = {'year': 2017, 'description': 'This is v2.0 of the VQA dataset.', 'contributor': 'VQA Team', 'version': '2.0', 'date_created': '2017-04-26 17:07:13', 'url': 'http://visualqa.org'}
#         dict_json['data_type'] = 'mscoco'
        with open(filename, 'w') as f:
            print(list(self.qns.keys()))
            json.dump(self.qns, f)

def helper_ans_string(answers, answers_conf = None):
    """ Helper function to generate output string of answers in the format of a list of dicts.
        list of dicts : [{"answer_id":answer_id, "answer":answer, "answer_conf": answer_conf},..]
        returns the processed list of dictionaries as a string.
    """
    list_of_dicts = []
    # Assume dominant answer is the first answer.
    
    if len(answers) == 0:
        raise 'No answers given. Pl check.'
        
    if answers_conf:
        if len(answers)!=len(answers_conf): raise 'Length of answers and answers_conf incompatible.'
    answers = answers[:10]
    for i, ans in enumerate(answers):
        dict_ans = {}
        dict_ans['answer_id'] = i+1
        dict_ans['answer'] = ans
        if answers_conf:
            dict_ans['answer_confidence'] = answers_conf[i]
        else:
            dict_ans['answer_confidence'] = 'yes'
        list_of_dicts.append(dict_ans)
            
    if len(answers)>10:
        raise Exception("Number of answers > 10. Given length: {}".format(len(answers)))
    elif len(list_of_dicts)<10:
        for i in range(len(answers),10):
            dict_ans = {}
            dict_ans['answer_id'] = i+1
            dict_ans['answer'] = answers[0]
            if answers_conf:
                dict_ans['answer_confidence'] = answers_conf[0]
            else:
                dict_ans['answer_confidence'] = 'yes'
            list_of_dicts.append(dict_ans)           
    return list_of_dicts

# TODO: add class functionality
def add_to_dataset(DataLoader, data):
    """ Adds all the questions and answer pairs from the data variable to the current Pandas Dataframes.
        Data format: {image_id:[(Question, Answer String)..]},{..}
        DataLoader : Class of data_loader
    """
    for k,v in data.items():
        image_id = k
        for (ques, ans) in v:
            qid = DataLoader.add_question(image_id,ques)
            DataLoader.add_answer(ans,qid, image_id)
                
#DataLoader = dataloader(config.question_train_path, config.answer_train_path)



    

