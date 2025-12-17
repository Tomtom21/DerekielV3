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
