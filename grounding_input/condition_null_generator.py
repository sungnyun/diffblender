import os 
import torch as th 



class BoxConditionInput:
    def __init__(self):
        self.set = False 

    def prepare(self, batch):
        self.set = True

        boxes = batch["values"]
        masks = batch["masks"]
        text_embeddings = batch["text_embeddings"]

        self.batch_size, self.max_box, self.embedding_len = text_embeddings.shape
        self.device = text_embeddings.device
        self.dtype = text_embeddings.dtype

        # return {"values": boxes, "masks": masks, "text_embeddings": text_embeddings}

    def get_null_input(self, batch, batch_size=None, device=None, dtype=None):
        assert self.set, "not set yet, cannot call this funcion"
        batch_size =  self.batch_size  if batch_size  is None else batch_size
        device = self.device if device is None else device
        dtype = self.dtype   if dtype  is None else dtype

        boxes = th.zeros(batch_size, self.max_box, 4).type(dtype).to(device) 
        masks = th.zeros(batch_size, self.max_box).type(dtype).to(device) 
        text_embeddings = th.zeros(batch_size, self.max_box, self.embedding_len).type(dtype).to(device) 

        batch["values"] = boxes
        batch["masks"] = masks
        batch["text_embeddings"] = text_embeddings

        return batch


class KeypointConditionInput:
    def __init__(self):
        self.set = False 

    def prepare(self, batch):
        self.set = True
        
        points = batch["values"] 
        masks = batch["masks"]

        self.batch_size, self.max_persons_per_image, _ = points.shape
        self.max_persons_per_image = int(self.max_persons_per_image / 17) 
        self.device = points.device
        self.dtype = points.dtype

        # return {"values": points, "masks": masks}

    def get_null_input(self, batch, batch_size=None, device=None, dtype=None):
        assert self.set, "not set yet, cannot call this funcion"
        batch_size =  self.batch_size  if batch_size  is None else batch_size
        device = self.device if device is None else device
        dtype = self.dtype   if dtype  is None else dtype

        points = th.zeros(batch_size, self.max_persons_per_image*17, 2).to(device) 
        masks = th.zeros(batch_size, self.max_persons_per_image*17).to(device) 

        batch["values"] = points
        batch["masks"] = masks

        return batch


class NSPVectorConditionInput:
    def __init__(self):
        self.set = False 

    def prepare(self, batch):
        self.set = True
        
        vectors = batch["values"] 
        masks = batch["masks"]

        self.batch_size, self.in_dim = vectors.shape
        self.device = vectors.device
        self.dtype = vectors.dtype

        # return {"values": vectors, "masks": masks}

    def get_null_input(self, batch, batch_size=None, device=None, dtype=None):
        assert self.set, "not set yet, cannot call this funcion"
        batch_size =  self.batch_size  if batch_size  is None else batch_size
        device = self.device if device is None else device
        dtype = self.dtype   if dtype  is None else dtype

        vectors = th.zeros(batch_size, self.in_dim).to(device) 
        masks = th.zeros(batch_size, 1).to(device) 

        batch["values"] = vectors
        batch["masks"] = masks

        return batch


class ImageConditionInput:
    def __init__(self):
        self.set = False

    def prepare(self, batch):
        self.set = True
        
        image = batch["values"]
        self.batch_size, self.in_channel, self.H, self.W = image.shape
        self.device = image.device
        self.dtype = image.dtype

        # return {"values": image}

    def get_null_input(self, batch, batch_size=None, device=None, dtype=None):
        assert self.set, "not set yet, cannot call this funcion"
        batch_size =  self.batch_size  if batch_size  is None else batch_size
        device = self.device if device is None else device
        dtype = self.dtype   if dtype  is None else dtype

        image = th.zeros(batch_size, self.in_channel, self.H, self.W).to(device) 
        masks = th.zeros(batch_size, 1).to(device)

        batch["values"] = image
        batch["masks"] = masks

        return batch

