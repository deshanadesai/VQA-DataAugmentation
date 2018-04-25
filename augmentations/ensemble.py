
class Templates:
    def __init__():
        pass
    
    def count_templates(obj1, pos, obj2):
        """Templates for questions of the Position + Counting type.
        """
        base_templates = ["How many {} are to the {} of {}".format(obj1, pos, obj2)]
        return random.choice(base_templates)
    
    def obj_recog_templates(obj1, pos):
        """Templates for questions of the Position + Object Recognition with Description type.
        """        
        base_templates = ["What is on the {} of the {}?".format(pos, obj1)]
        return random.choice(base_templates)
    
    def pos_templates(obj1, obj2):
        """Templates for questions of the Position (Descriptive) type.
        """        
        base_templates = ["Where is the {} placed in relation to the {}".format(obj2, obj1),
                          "On what side of the {} is the {}?".format(obj1, obj2)]
        return random.choice(base_templates)
    
    def yes_or_no_templates(obj1, pos, obj2):
        """Templates for questions of the Position Yes/No type.
        """        
        base_templates = ["Is there a {} on the {} side of the {}?".format(obj1,pos,obj2),
                          "Is a {} placed on the {} side of the {}?".format(obj1,pos,obj2)]
        return random.choice(base_templates)    

    def color_templates(obj1, pos, obj2):
        """Templates for questions of the Position Yes/No type.
        """        
        base_templates = ["What is the color of the {} on the {} of {}?".format(obj1,pos,obj2)]
        return random.choice(base_templates)    

class Ensemble:
    def __init__():
        self.templates = Templates()
        
    def adjective_obj(adj1, adj2, obj1, obj2):
        if adj1 and adj2:
            obj1_string = adj1 + " " + obj1
            obj2_string  = adj2 + " " + obj2
            
        elif adj1:
            obj1_string = adj1 + " " + obj1
            obj2_string = obj2
            
        elif adj2: 
            obj1_string = obj1
            obj2_string = adj2 + " " + obj2
            
        else:
            obj1_string = obj1
            obj2_string = obj2
        return obj1_string, obj2_string
                
    def gen_obj_question(obj1,obj2,pos,adj1):
        obj1, obj2 = adjective_obj(adj1,adj2,obj1,obj2)        
        q = self.templates.obj_recog_templates(obj1, pos)
        return (q,obj2)
    
    def gen_yes_or_no(obj1,obj2,pos,adj1,ans, adj2 = None): 
        obj1, obj2 = adjective_obj(adj1,adj2,obj1,obj2)
        q = self.templates.yes_or_no_templates()        
        return (q,ans)
        
    def gen_pos_question(obj1,obj2,pos,adj1 = None, adj2 = None):
        obj1, obj2 = adjective_obj(adj1,adj2,obj1,obj2)
        q = self.templates.pos_templates(obj1, obj2)
        return (q,pos)

    def gen_count_question(obj1, pos, obj2, count, adj1 = None, adj2 = None):
        obj1, obj2 = adjective_obj(adj1,adj2,obj1,obj2)
        q = self.templates.count_templates(obj1, pos, obj2)
        return (q, count)       
    
    def gen_color_question(obj1, pos, obj2, color, adj1 = None, adj2 = None):
        obj1, obj2 = adjective_obj(adj1,adj2,obj1,obj2)        
        q = self.templates.color_templates(obj1, pos, obj2)
        return (q, color)        
    
            
           