# TraficViolence_RedLightRule

**Dataset**
Crawl data video for Dong Hoi Smart City App

9 classes: bike, bus, car,motobike, truck, green_light, red_light, yellow_light stop_line

| Label | Number of image |
|--------------|-------|
| 9 | 2564 |

 <img src="https://github.com/Lecongquochuy/TraficViolence_RedLightRule/blob/main/Results/split_data.png">

**Propose System**

 <img src="https://github.com/Lecongquochuy/TraficViolence_RedLightRule/blob/main/Results/Psystem.png">
 
   The proposed system is divided into three main stages: object detection,
vehicle tracking and violation detection. The object detection phase will identify and determine the
location of vehicles, traffic lights, and stop lines. The vehicle’s position is determined by the endpoint
of the bounding box.In the next stage, track multiple vehicle in videos by associating object detections
across frames using a combination of motion and appearance information. In violation detection,
based on information about the vehicle’s position and the stop line, violation will be detected

 **- Using yolov8 to dectect object**
 
 <img src="https://github.com/Lecongquochuy/TraficViolence_RedLightRule/blob/main/train/train_batch0.jpg">
 <img src="https://github.com/Lecongquochuy/TraficViolence_RedLightRule/blob/main/Results/results.png">

**- Using SORT to object tracking**

 <img src="https://github.com/Lecongquochuy/TraficViolence_RedLightRule/blob/main/Results/SORT.jpg">

 **- Violation detection**
  <img src="https://github.com/Lecongquochuy/TraficViolence_RedLightRule/blob/main/Results/result_violated.png">

**RESULTS**
  
  ![Video Demo](https://github.com/Lecongquochuy/TraficViolence_RedLightRule/blob/main/Result_video/output_video1.gif)

