```mermaid
graph TD;
    %% Legend
    subgraph Legend
        Video_Frame((Input Frame))
        ML_Model[(ML Model)]
        Processing_Module{{Processing Module}}
    end

    %% Inputs
    Rear_Input_Frame((Rear Camera Frame))
    Driver_Input_Frame((Driver-Facing Frame))
    GPS_Speed_Data((GPS Speed Data))
    Gyroscope((Gyroscope))

    %% Rear camera pipeline
    Rear_Input_Frame --> Frame_Handler{{Frame Handler}}
    Frame_Handler --> YOLO[(YOLO Model)]
    Frame_Handler --> Lane_Detection[(Lane Detection)]

    %% Tracking & data association
    YOLO -- Detected objects --> Data_Association{{Data Association}}
    Lane_Detection -- Lane Masks--> Data_Association
    Data_Association --> Object_States>Updated Object States]
    Depth_Est{{Depth Estimation Module}} --> Object_States
    Vehicle_Type_Est --> Object_States
    GPS_Speed_Data --> Object_States
    OCR[(OCR<br>For Speedsign Reading)] -- After 3-5 consistent frames, Laplacian, then OCR --> Object_States

    %% Driver attention monitoring
    Driver_Input_Frame --> Face_Track[(Face Tracking Model)]
    Face_Track --> Driver_State>Driver State]
    GPS_Speed_Data --> Driver_State

    %% Visualization and output
    Object_States --> Renderer3D{{3D Visualization / HUD Renderer}}
    Driver_State --> Renderer3D
    Object_States --> Safety_Eval{{Safety Evaluation Module}}
    Driver_State --> Safety_Eval
    Safety_Eval --> Renderer3D
    GPS_Speed_Data --> Safety_Eval
    GPS_Speed_Data --> Renderer3D
    Renderer3D --> Display((Display))
    Safety_Eval --> Display
    Safety_Eval --> Sound((Sound))
    Gyroscope --> Parking_Transform{{Parking Image Transform}}
    Parking_Transform --> Display
    Gyroscope -- Leveling --> Display

    %% Notes
    %% Improved color contrast
    classDef input fill:#f6f8fa,stroke:#999,color:#111;
    classDef model fill:#ffb347,stroke:#333,color:#fff;
    classDef module fill:#80cfa9,stroke:#333,color:#fff;
    classDef output fill:#4db8ff,stroke:#333,color:#fff;
    class Rear_Input_Frame,Driver_Input_Frame,GPS_Speed_Data,Gyroscope input;
    class YOLO,Lane_Detection,Vehicle_Type_Est,OCR,Driver_Attn,Face_Track model;
    class Frame_Handler,Tracker,Data_Association,Depth_Est,Renderer3D,Safety_Eval,Parking_Transform module;
    class Object_States,System_Output,Driver_State,Display,Sound output;
```

## Pipeline
The pipeline for Derekiel V3 is as follows:
1. **YOLO/LaneNet model inference**
2. **Associate bounding boxes with a lane**
   1. Regardless of the object, this helps split up the association work so that we aren't trying to associate bboxes across the whole image, just one "lane".
3. **Associating bboxes with known objects** (Objects will be handled differently based on the IoU info)
   1. New detections, no IoU match
      1. New object is created in object states
   2. Object with no Δcx/Δcy
      1. Run regular IoU to find where the object is. Establishes Δcx/Δcy.
   3. Object with Δcx/Δcy
      1. Run standard IoU OR enhanced IoU (predicts future position of bbox using Δcx/Δcy), also updates Δcx/Δcy
   4. Object without YOLO detection
      1. Increment missed_frames value
      2. Delete objects based on variable missed_frames value
   5. Other things to remember
      1. Associate by lane first
         1. Objects typically won't change lanes in the short term
      2. Make depth/velocity estimates
         1. This requires that we keep a memory of the raw bbox size info, potentially 3-5 frames of info
      3. Extrapolation (Δcx/Δcy) is for helping to predict where the bounding box may be in the future. Not absolutely necessary
4. Can plan/sort/filter based on vehicle objects (or just TrackObjects) on lane or other properties like distance
5. Potentially expand object tracking to other objects
   1. Stoplights (knowing what lane they're for)
   2. Speed signs
      1. Save each image in the object
      2. Track which one is most clear, try OCR
         3. Uses Laplacian

## Classes
1. TrackedObject
   1. lane: 
   2. object_type: string or VehicleType enum
   3. missed_frames: int
   4. bbox_width: list of ints
   5. bbox_height: list of ints
   6. bbox_cx: list of ints
   7. bbox_cy: list of ints
   8. age: int
   9. Will require helper parent classes that can help dervice depth, keep the focal lengths
