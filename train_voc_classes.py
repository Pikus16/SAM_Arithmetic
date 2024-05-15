#from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np 
from torch.utils.data import Dataset, Subset
from transformers import SamProcessor, SamModel
from PIL import Image
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
import os 
import click


voc_class_names = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
] # 255 is ignore

def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]
  return bbox

class SAMDataset(Dataset):
  def __init__(self, dataset, processor, desired_class_id, use_bounding_box = False):
    self.dataset = dataset
    self.processor = processor
    self.desired_class_id = desired_class_id
    self.use_bounding_box = use_bounding_box

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    #image = item["image"]
    #ground_truth_mask = np.array(item["label"])
    image, ground_truth_mask = item
    image = image.resize((1024,1024), Image.LANCZOS)
    ground_truth_mask = ground_truth_mask.resize((256,256), Image.NEAREST)
    ground_truth_mask = np.array(ground_truth_mask)

    ground_truth_mask[ground_truth_mask != self.desired_class_id] = 0
    ground_truth_mask[ground_truth_mask == self.desired_class_id] = 1

    # get bounding box prompt
    if self.use_bounding_box:
      prompt = [[get_bounding_box(ground_truth_mask)]]
    else:
      prompt = None
    # prepare image and prompt for the model
    inputs = self.processor(image,
                            input_boxes=prompt,
                            return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask
    
    return inputs

def setup_model_name(class_name, use_box, train_mask, train_prompt, train_vision):
  if use_box:
    model_name = f'voc_trained/box/'
  else:
    model_name = f'voc_trained/nobox/'

  if train_mask:
    model_name = os.path.join(model_name, 'mask')
  elif train_prompt:
    model_name = os.path.join(model_name, 'prompt')
  elif train_vision:
    model_name = os.path.join(model_name, 'vision')
  else:
    assert False
  
  if not os.path.exists(model_name):
    os.mkdir(model_name)

  model_name = os.path.join(model_name, f'{class_name}.pth') 
  return model_name 

def get_dataset_of_class(desired_class_id, path_to_dataset='vocsegmentation', image_set='train'):
  # VOC (Airplane)
  ds = torchvision.datasets.VOCSegmentation(path_to_dataset, image_set=image_set)
  inds = []
  for i,d in enumerate(ds):
      ct = np.unique(d[1])
      if desired_class_id in ct:
        inds.append(i)
  ds = Subset(ds, inds)
  return ds

def get_dataset(use_box, desired_class_id):
  processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
  dataset = get_dataset_of_class(desired_class_id=desired_class_id, image_set='train')
  train_dataset = SAMDataset(dataset=dataset, processor=processor, desired_class_id=desired_class_id, use_bounding_box=use_box)
  return train_dataset

def get_model(train_mask, train_prompt, train_vision):
  model = SamModel.from_pretrained("facebook/sam-vit-base")

  # make sure we only compute gradients for mask decoder
  for name, param in model.named_parameters():
    if name.startswith("vision_encoder"):
      param.requires_grad_(train_vision)
    elif name.startswith("prompt_encoder"):
      param.requires_grad_(train_prompt)
    elif name.startswith("mask_decoder"):
      param.requires_grad_(train_mask)
    else:
      assert False

  num_trainable = 0
  trainable = []
  for name, param in model.named_parameters():
    if param.requires_grad:
      trainable.append(param)
      num_trainable += 1
  print(f'Number of trainable params: {num_trainable}')
  return model, trainable


@click.command()
@click.option('--use-box', is_flag=True, help='Flag to use box.')
@click.option('--train-mask', is_flag=True, help='Flag to train mask.')
@click.option('--train-prompt', is_flag=True, help='Flag to train prompt.')
@click.option('--train-vision', is_flag=True, help='Flag to train vision.')
@click.option('--num-epochs', default=50, help='Number of epochs to train.')
def run(use_box, train_mask, train_prompt, train_vision, num_epochs):
  print(f'Vision: {train_vision}, Mask: {train_mask}, Prompt: {train_prompt}, Use Box {use_box}, Num Epochs {num_epochs}')

  for desired_class_id in range(1, len(voc_class_names)):
    class_name = voc_class_names[desired_class_id]
    model_name = setup_model_name(class_name, use_box, train_mask, train_prompt, train_vision )

    train_dataset = get_dataset(use_box, desired_class_id)
    print(class_name, len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=8)

    model, trainable = get_model(train_mask, train_prompt, train_vision)

    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(trainable, lr=1e-5, weight_decay=0)

    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            if use_box:
                input_boxes = batch["input_boxes"].to(device)
            else:
                input_boxes = None

            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=input_boxes,
                            multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
        if epoch > 0 and epoch % 5 == 0:
            torch.save(model.state_dict(), model_name)

    torch.save(model.state_dict(), model_name)
    model.to('cpu')
    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
  run()