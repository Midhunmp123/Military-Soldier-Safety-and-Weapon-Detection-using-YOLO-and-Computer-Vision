import subprocess

def train():

    cmd = [
'python', '-m', 'torch.distributed.run', '--nproc_per_node=1',
'train.py',
'--img', '640',
'--batch', '16',
'--epochs', '50',
'--data', 'dataset/dataset.yaml',  # Updated path to dataset yaml
'--weights', 'yolov5s.pt',
'--project', 'runs/train',
'--name', 'military_yolo'
]
    subprocess.run(cmd)

if __name__ == '__main__':
    train()
