# Livestock-Management Project Brief
My final year project is titled “Livestock Management with OpenCV and a UAV (Drone)”. The goal
of this project was to analyse the first-person view from a UAV and process it in real time using
OpenCV to create functionality such as detecting and counting objects in motion.

I decided to take this project goal and base it around the revolutionary use of drones in agriculture.
During this project, I focused on the recognition and detection of livestock animals such as cattle
and sheep. Combining techniques of machine learning, image processing and python
programming, I was able to train my own object detection models to recognise livestock. With
the aid of the OpenCV library I was able to apply these object detection models to the eyes of a
drone while highlighting and counting each of the livestock animals in frame.

With the autonomous flight feature of a drone and an accurate object detection model, a project
such as this could prove useful in livestock farming as it would simplify the process of, e.g., keeping
count on various livestock animals and general monitoring of livestock.

## Important NOTE for this repository:
After training the object detection model (YOLO) on a custom image dataset, the custom model's final weights that are produced after training are too large in size to be uploaded to the github repository.  
Therefore, the files for the object training process of this project are stored in my google drive, where google colab notebooks are used to train custom models using Google's free GPU resources and the subsequent custom model weights are saved to my drive.
Access to these files can obtained on request: lorcancreedon1@gmail.com
