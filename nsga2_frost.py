import subprocess
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import time
import os.path
'''
0:dropout
1:gdroput
2:dropblock
'''
class model:
    count=0
    def __init__(self, drop_rate: float, dropout_type: int, num_sample: int):
        self.drop_rate=str(drop_rate)
        self.dropout_type=dropout_type
        self.num_sample=str(num_sample)
        
    def run(self):
        model.count+=1
        old_time = time.time()
        cfg_list=["yolov5s-dropout.yaml","yolov5s-gdropout.yaml","yolov5s-dropblock.yaml"]
        print("running yolov5...")
        
        subprocess.call(['python', 'val.py','--cfg',cfg_list[self.dropout_type],
        '--batch','4','--data','coco.yaml','--imgsz','640','--iou-thres','0.6',
        '--num_samples',self.num_sample,'--conf-thres','0.5','--new_drop_rate',
        self.drop_rate,'--corruption_num','8','--severity','2'],stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT)
        print("evaluating PDQ and mAP...")
                    
 
        subprocess.call(['python', 'pdq_evaluation/evaluate.py','--test_set','coco',
        '--gt_loc','../datasets/coco/annotations/instances_val2017.json',
        '--det_loc','dets_converted_exp_0.5_0.6.json',
        '--save_folder','output','--num_workers','16'],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        
        current_time = time.time()
        ti=(current_time-old_time)/60   
        data={}     
        with open(r"./output/scores.txt") as f:
            for line in f.readlines():
                if line:
                    name,value=line.strip().split(':',1)
                    data[name]=float(value) 
        print("PDQ: {0:4f} mAP: {1:4f} running time:{2:.03}min".format(data['PDQ'],data['mAP'],ti))
        print('evaluation time:{0}\n'.format(model.count))

        return [data['PDQ'],data['mAP']]


class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, xl=np.array([0,0,2]), xu=[0.5,2,20])

    def _evaluate(self, x, out, *args, **kwargs):
        print(x)
        
        Model=model(x[0],x[1],x[2])
        output=Model.run()
        f1,f2=output[0]*(-1),output[1]*(-1)  
        out["F"] = np.column_stack([f1,f2])

mask = ["real", "int", "int"]

sampling = MixedVariableSampling(mask, {
    "real": get_sampling("real_random"),
    "int": get_sampling("int_random")
})

crossover = MixedVariableCrossover(mask, {
    "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
    "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
})

mutation = MixedVariableMutation(mask, {
    "real": get_mutation("real_pm", eta=3.0),
    "int": get_mutation("int_pm", eta=3.0)
})
problem = MyProblem()

algorithm = NSGA2(pop_size=50,sampling=sampling,crossover=crossover,mutation=mutation,eliminate_duplicates=True,)

filename='./checkpoint.npy'
if os.path.isfile(filename):
    checkpoint, = np.load("checkpoint.npy", allow_pickle=True).flatten()
    print("Loaded Checkpoint:", checkpoint)
    checkpoint.has_terminated = False
    
    res = minimize(problem,
               checkpoint,
               ('n_gen', 1),
               seed=1,
               copy_algorithm=False,
               verbose=True)
    print("the final designspace:")
    print(res.X)
    print("the final map and pdq:")
    print(res.F)
    np.save("checkpoint", checkpoint)
else:
    res = minimize(problem,
               algorithm,
               ('n_gen', 1),
               seed=1,
               copy_algorithm=False,
               verbose=True)
    print("the final designspace:")
    print(res.X)
    print("the final map and pdq:")
    print(res.F)
    np.save("checkpoint", algorithm)
