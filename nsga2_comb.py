import subprocess
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import time
'''
0:dropout-None
1:gdroput-None
2:dropblock-None
3:dropout-gdropout
4:dropout-dropblock
5:gdropout-dropblock
'''
class model:
    def __init__(self, drop_rate:float, second_dropout_rate:float, dropout_type:int):
        self.drop_rate=str(drop_rate)
        self.second_dropout_rate=str(drop_rate)
        self.dropout_type=dropout_type
        
    def run(self):
        old_time = time.time()
        cfg_list=["yolov5s-dropout.yaml","yolov5s-gdropout.yaml","yolov5s-dropblock.yaml","yolov5s-dg.yaml","yolov5s-db.yaml","yolov5s-gb.yaml"]
        print("running yolov5...")
        subprocess.call(['python', 'val.py','--cfg',cfg_list[self.dropout_type],'--batch','8','--data','coco.yaml','--imgsz','640','--iou-thres','0.6','--num_samples','8','--conf-thres','0.5','--new_drop_rate',self.drop_rate,'--second_drop_rate',self.second_dropout_rate],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        print("evaluating PDQ and mAP...")
        subprocess.call(['python', 'pdq_evaluation/evaluate.py','--test_set','coco','--gt_loc','/home/chen/Project/datasets/coco/annotations/instances_val2017.json','--det_loc','/home/chen/Project/stochastic-yolov5/dets_converted_exp_0.5_0.6.json','--save_folder','output','--num_workers','16'],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        data={}
        data_snow={}
        data_frost={}
        data_fog={}        
        with open(r"./output/scores.txt") as f:
            for line in f.readlines():
                if line:
                    name,value=line.strip().split(':',1)
                    data[name]=float(value) 
        print("PDQ: {0:4f} mAP: {1:4f}".format(data['PDQ'],data['mAP']))
        
        #snow
        print("running yolov5 on corruption snow...")
        subprocess.call(['python', 'val.py','--cfg',cfg_list[self.dropout_type],'--batch','8','--data','coco.yaml','--imgsz','640','--iou-thres','0.6','--num_samples','8','--conf-thres','0.5','--new_drop_rate',self.drop_rate,'--corruption_num','7','--severity','3','--second_drop_rate',self.second_dropout_rate],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        subprocess.call(['python', 'pdq_evaluation/evaluate.py','--test_set','coco','--gt_loc','/home/chen/Project/datasets/coco/annotations/instances_val2017.json','--det_loc','/home/chen/Project/stochastic-yolov5/dets_converted_exp_0.5_0.6.json','--save_folder','output','--num_workers','16'],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        with open(r"./output/scores.txt") as f:
            for line in f.readlines():
                if line:
                    name,value=line.strip().split(':',1)
                    data_snow[name]=float(value)
        print("PDQ: {0:4f} mAP: {1:4f}".format(data_snow['PDQ'],data_snow['mAP']))
        
        #frost
        print("running yolov5 on corruption frost...")
        subprocess.call(['python', 'val.py','--cfg',cfg_list[self.dropout_type],'--batch','8','--data','coco.yaml','--imgsz','640','--iou-thres','0.6','--num_samples','8','--conf-thres','0.5','--new_drop_rate',self.drop_rate,'--corruption_num','8','--severity','3','--second_drop_rate',self.second_dropout_rate],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        subprocess.call(['python', 'pdq_evaluation/evaluate.py','--test_set','coco','--gt_loc','/home/chen/Project/datasets/coco/annotations/instances_val2017.json','--det_loc','/home/chen/Project/stochastic-yolov5/dets_converted_exp_0.5_0.6.json','--save_folder','output','--num_workers','16'],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        with open(r"./output/scores.txt") as f:
            for line in f.readlines():
                if line:
                    name,value=line.strip().split(':',1)
                    data_frost[name]=float(value)
        print("PDQ: {0:4f} mAP: {1:4f}".format(data_frost['PDQ'],data_frost['mAP']))
        
        #fog
        print("running yolov5 on corruption fog...")
        subprocess.call(['python', 'val.py','--cfg',cfg_list[self.dropout_type],'--batch','8','--data','coco.yaml','--imgsz','640','--iou-thres','0.6','--num_samples','8','--conf-thres','0.5','--new_drop_rate',self.drop_rate,'--corruption_num','9','--severity','3','--second_drop_rate',self.second_dropout_rate],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        subprocess.call(['python', 'pdq_evaluation/evaluate.py','--test_set','coco','--gt_loc','/home/chen/Project/datasets/coco/annotations/instances_val2017.json','--det_loc','/home/chen/Project/stochastic-yolov5/dets_converted_exp_0.5_0.6.json','--save_folder','output','--num_workers','16'],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        with open(r"./output/scores.txt") as f:
            for line in f.readlines():
                if line:
                    name,value=line.strip().split(':',1)
                    data_fog[name]=float(value) 
        print("PDQ: {0:4f} mAP: {1:4f}".format(data_fog['PDQ'],data_fog['mAP']))     
        r_PDQ=(data_snow['PDQ']+data_frost['PDQ']+data_fog['PDQ'])/data['PDQ']/3
        r_mAP=(data_snow['mAP']+data_frost['mAP']+data_fog['mAP'])/data['mAP']/3
        print("the final PDQ: {0:4f}\nmAP: {1:4f}\nr_PDQ: {2:4f}\nr_mAP {3:4f}".format(data['PDQ'],data['mAP'],r_PDQ,r_mAP))
        
        current_time = time.time()
        run_time=(current_time - old_time)/60
                
        print("running time:{:.3}min\n".format(run_time))

        return [data['PDQ'],data['mAP'],r_PDQ,r_mAP]


class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=4, xl=np.array([0, 0, 0]), xu=[0.3,0.3,5])

    def _evaluate(self, x, out, *args, **kwargs):
        print(x)
        
        Model=model(x[0],x[1],x[2])
        output=Model.run()
        f1,f2,f3,f4=output[0]*(-1),output[1]*(-1),output[2]*(-1),output[3]*(-1)    
        out["F"] = np.column_stack([f1,f2,f3,f4])

mask = ["real", "real", "int"]

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
'''
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

plot = Scatter()
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()
plot.save("res.png")
np.save("checkpoint", algorithm)
'''
resume=1
if resume==1:
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
    plot = Scatter()
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.show()
    plot.save("res.png")
    np.save("checkpoint", checkpoint)
