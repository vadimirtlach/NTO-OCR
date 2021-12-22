from torch import DataLoader
import numpy as np
import string 
import cv2


class Dataset:
    russian_lower_letters = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    russian_upper_letters = list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯ")
    english_lower_letters = list(string.ascii_lowercase)
    english_upper_letters = list(string.ascii_uppercase)
    digits = list(string.digits)
    puntuations = list(string.punctuation + " ")
    special_tokens = ["<BOS>", "<PAD>", "<EOS>", "<OOV>"]
    characters = special_tokens + russian_lower_letters + russian_upper_letters + english_lower_letters + english_upper_letters + digits + puntuations
    classes = {idx: character for idx, character in enumerate(characters)}
    reversed_classes = {character: idx for idx, character in enumerate(characters)}
    
    oov_token_index = reversed_classes["<OOV>"]
    bos_token_index = reversed_classes["<BOS>"]
    pad_token_index = reversed_classes["<PAD>"]
    eos_token_index = reversed_classes["<EOS>"]
    
    def __init__(self, pathes, labels=None, transforms=None, max_length=25):
        self.pathes = pathes
        self.labels = labels
        self.transforms = transforms
        self.max_length = max_length
        
    def __len__(self):
        return len(self.pathes)
    
    @staticmethod
    def text2label(text):
        label = []
        for _ in list(text):
            encoded = Dataset.reversed_classes.get(str(_), Dataset.oov_token_index)
            label.append(encoded)
            
        label = np.array(label)
        return label
    
    @staticmethod
    def label2text(label):
        text = []
        for _ in label:
            decoded = Dataset.classes.get(int(_), Dataset.classes[Dataset.oov_token_index])
    
            if decoded not in Dataset.special_tokens:
                text.append(decoded)
        
        text = "".join(text)
        return text
    
    @staticmethod
    def add_special_tokens(label, max_length):
        label = list(label)
        length = len(label)
        assert max_length >= length, f"'max_length' must be equal or greater than input sequence's length, but input sequence's length ({length}) > max length ({max_length})"
        num_paddings = max_length - length
        label = [Dataset.bos_token_index] + label + [Dataset.pad_token_index] * num_paddings  + [Dataset.eos_token_index]
        label = np.array(label)
        return label
    
    def __getitem__(self, index):
        path = self.pathes[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image,axis=-1)
        
        if self.transforms:
            image = self.transforms(image=image)["image"]
        
        if self.labels is not None:
            label = self.labels[index]
            label = Dataset.text2label(label)
            label = Dataset.add_special_tokens(label, max_length=self.max_length)
            return image, label
        
        return image
    
    @staticmethod
    def create_loader(pathes, labels=None, transforms=None, max_length=25, batch_size=16, pin_memory=False, num_workers=0, shuffle=False, drop_last=False):
        dataset = Dataset(pathes=pathes, 
                          labels=labels, 
                          max_length=max_length,
                          transforms=transforms)
        
        loader = DataLoader(dataset=dataset, 
                            batch_size=batch_size, 
                            pin_memory=pin_memory, 
                            num_workers=num_workers, 
                            shuffle=shuffle, 
                            drop_last=drop_last)
        
        return dataset, loader
    
    @staticmethod
    def collate_batch(batch, inputs_device="cpu", labels_device="cpu"):
        if len(batch) > 1:
            inputs, labels = batch
            inputs = inputs.to(inputs_device).float()
            labels = labels.to(labels_device).float()
            
            return inputs, labels
        else:
            return batch.to(inputs_device)