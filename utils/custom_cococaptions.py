from PIL import Image
import os
import os.path
import torch
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms
from pycocotools.coco import COCO

class CocoCaptions(VisionDataset):
	# Returns images at two different resolutions
	def __init__(self, root, annFile, transform_gan=None, transform_vgg=None,  target_transform=None, transforms=None, num_captions=5):
		super(CocoCaptions, self).__init__(root, None, None, None)
		self.coco = COCO(annFile)
		self.transform_gan = transform_gan
		self.transform_vgg = transform_vgg
		self.num_captions = num_captions

		all_ids = sorted(self.coco.imgs.keys())
		# Keep only images with at least `num_captions` captions
		self.ids = [img_id for img_id in all_ids
		            if len(self.coco.getAnnIds(imgIds=img_id)) >= self.num_captions]

		dropped = len(all_ids) - len(self.ids)
		print(f"CocoCaptions: kept {len(self.ids)} images, "
		      f"dropped {dropped} with fewer than {self.num_captions} captions")

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: Tuple (image, target). target is a list of captions for the image.
		"""
		coco = self.coco
		img_id = self.ids[index]

		# sort the annotations by their id to ensure consistent ordering
		ann_ids = sorted(coco.getAnnIds(imgIds=img_id))
		anns = coco.loadAnns(ann_ids)
		target = [ann['caption'] for ann in anns][:self.num_captions] # get the first `num_captions` captions

		path = coco.loadImgs(img_id)[0]['file_name']
		img = Image.open(os.path.join(self.root, path)).convert('RGB')

		if self.transform_gan is not None:
			img_gan = self.transform_gan(img)

		if self.transform_vgg is not None:
			img_vgg = self.transform_vgg(img)	

		return img_gan, img_vgg, target


	def __len__(self):
		return len(self.ids)

