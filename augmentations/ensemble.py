
class Templates:
    def __init__(self):
        pass
    
    def count_templates(self, obj1, pos, obj2):
        """Templates for questions of the Position + Counting type.
        """
        base_templates = ["How many {} are to the {} of {}".format(obj1, pos, obj2)]
        return random.choice(base_templates)
    
    def obj_recog_templates(self, obj1, pos):
        """Templates for questions of the Position + Object Recognition with Description type.
        """        
        base_templates = ["What is on the {} of the {}?".format(pos, obj1)]
        return random.choice(base_templates)
    
    def pos_templates(self, obj1, obj2):
        """Templates for questions of the Position (Descriptive) type.
        """        
        base_templates = ["Where is the {} placed in relation to the {}".format(obj2, obj1),
                          "On what side of the {} is the {}?".format(obj1, obj2)]
        return random.choice(base_templates)
    
    def yes_or_no_templates(self, obj1, pos, obj2):
        """Templates for questions of the Position Yes/No type.
        """        
        base_templates = ["Is there a {} on the {} side of the {}?".format(obj1,pos,obj2),
                          "Is a {} placed on the {} side of the {}?".format(obj1,pos,obj2)]
        return random.choice(base_templates)    

    def color_templates(self, obj1, pos, obj2):
        """Templates for questions of the Position Yes/No type.
        """        
        base_templates = ["What is the color of the {} on the {} of {}?".format(obj1,pos,obj2)]
        return random.choice(base_templates)    

class Ensemble:
    def __init__(self):
        self.templates = Templates()
        
    def adjective_obj(self, adj1, adj2, obj1, obj2):
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
                
    def gen_obj_question(self, obj1,obj2,pos,adj1):
        obj1, obj2 = adjective_obj(adj1,adj2,obj1,obj2)        
        q = self.templates.obj_recog_templates(obj1, pos)
        return (q,obj2)
    
    def gen_yes_or_no(self, obj1,obj2,pos,adj1,ans, adj2 = None): 
        obj1, obj2 = adjective_obj(adj1,adj2,obj1,obj2)
        q = self.templates.yes_or_no_templates()        
        return (q,ans)
        
    def gen_pos_question(self, obj1,obj2,pos,adj1 = None, adj2 = None):
        obj1, obj2 = adjective_obj(adj1,adj2,obj1,obj2)
        q = self.templates.pos_templates(obj1, obj2)
        return (q,pos)

    def gen_count_question(self, obj1, pos, obj2, count, adj1 = None, adj2 = None):
        obj1, obj2 = adjective_obj(adj1,adj2,obj1,obj2)
        q = self.templates.count_templates(obj1, pos, obj2)
        return (q, count)       
    
    def gen_color_question(self, obj1, pos, obj2, color, adj1 = None, adj2 = None):
        obj1, obj2 = adjective_obj(adj1,adj2,obj1,obj2)        
        q = self.templates.color_templates(obj1, pos, obj2)
        return (q, color)   
    
    def generate_questions(self, image_id):
        pos = get_graph(image_id)
        
        # pick an object pair from the graph obj1, obj2. 
        # Let the annotation id be stored in obj1, obj2.
        
        color_obj1 = get_color_from_anno(obj_ann_1, img)
        color_obj2 = get_color_from_anno(obj_ann_2, img)
        
        # find count and pos of objs
        
        # Go through each object in the graph sorted by size with a probability.
        # If object has multiple answers on the <pos>, proceed.
        # For multiple objects (obj2) - get name, supercat, count, colors c, size of each of the names.
        # Pick an obj2 at random.
        # Question is what is needed to discriminate a particular obj2 from the rest of obj2's?
        # Can we build a graph? 
        # Decision tree with gini entropy?
        # Take the path which crosses off most other things.
        # get things with blue color. (random from list of colors c).
        # If there are multiple different types of objects that are blue, (filter the rest)
        # check if counts of these two cats are different (if yes, filtering the other like this) [max: 5]
        # If there are still multiple possible answers, 
        # check if can be differentiated by size or supercat.
        # if still multiple answers, fuck it. go back and try with another obj2/direction.
        
        # pick a direction based on where there are more objects.
        for obj in pos:
            direction = obj[direction].max()
        
        # direction and obj1 are picked.
        color_obj1 = get_color_from_anno(obj_ann_1, img)
        
        # pick an obj2.
        obj2 = random.choice(pos[obj1])
        
        # TODO: what if there are multiple obj1 of the same type?
        build_graph(obj1, obj2)
        
        
    def build_graph(obj1, obj2, colors, counts, supercats, sizes):
        
    
            
           