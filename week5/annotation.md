### Annotating images 

To annotate images, we will be using the labelImg package. For installation and usage instruction, please refer to [LabelImg github repo](https://github.com/tzutalin/labelImg)

Once you have collected all the images to be used to train your model (ideally more than 100 per class), place them inside a folder e.g. balloon_dataset. It is good practice to name your files with a sequence number for easy processing later on, e.g. balloon-1.jpg, balloon-2.jpg, etc. Choose PascalVOC format for your annotations. Start annotating, and you should have an .xml file generated for each of the image you annotate. 

Zip up all the files inside the directory (e.g. balloon_dataset) and upload the zip file to your cloud VM. Follow the instructions given in the custom_training_with_tfod_api.md file to generate appropriate train and validation set.

Copy all images, together with their corresponding `*.xml` files, and place them inside the ``images`` folder and ``annotations`` folder of your project data directory (e.g. data/images, data/annotations)
