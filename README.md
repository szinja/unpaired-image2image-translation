# CLIP-Enhanced CycleGAN for Unpaired Image-to-Image Translation

## Overview

This project implements an unpaired image-to-image translation model based on the CycleGAN architecture, enhanced with OpenAI's CLIP model for semantic guidance. Instead of traditional discriminators, CLIP is used to evaluate the semantic consistency between the generated images and target text descriptions, guiding the generators to produce more contextually relevant translations.

This implementation focuses on the Monet paintings to real photos translation task (Monet2Photo).

## Architecture

The core architecture consists of two generators (`G_A2B` and `G_B2A`) similar to standard CycleGAN. Key differences include:

* **No Discriminators:** The standard patch-based discriminators are removed.
* **CLIP Guidance:** A pre-trained CLIP model calculates a semantic loss based on the similarity between generated images and target text prompts (e.g., "a photo of a landscape", "a painting by Monet").
* **Loss Functions:** The model is trained using a combination of:
    * Cycle Consistency Loss
    * CLIP Semantic Loss
    * Identity Loss

```mermaid
graph LR
    subgraph "Domain A (e.g., Monet Paintings)"
        direction LR
        xA[Real_A]
    end

    subgraph "Domain B (e.g., Photos)"
        direction LR
        yB[Real_B]
    end

    subgraph "Generator A to B"
        direction TB
        G[Generator_G_A2B]
    end

    subgraph "Generator B to A"
        direction TB
        F[Generator_F_B2A]
    end

    subgraph "CLIP Model"
        direction TB
        CLIP[CLIP_Model]
    end

    subgraph "Loss Components"
        direction TB
        CycleLoss[Cycle_Consistency_Loss]
        CLIPLoss[CLIP_Semantic_Loss]
        IdentLoss[Identity_Loss_Optional]
    end

    subgraph "Text Prompts"
        direction TB
        PromptB["Prompt: 'a_photo_of_a_landscape...'"]
        PromptA["Prompt: 'a_Monet_painting...'"]
    end

    %% Forward Pass A -> B -> A
    xA -- "Input" --> G -- "Generates" --> y_hatB[Fake_B];
    y_hatB -- "Input" --> F -- "Generates" --> x_hatA[Cycled_A];

    %% Forward Pass B -> A -> B
    yB -- "Input" --> F -- "Generates" --> x_hatA2[Fake_A];
    x_hatA2 -- "Input" --> G -- "Generates" --> y_hatB2[Cycled_B];

    %% Cycle Consistency Loss Calculation
    x_hatA -- "Compare" --> CycleLoss;
    xA -- "Compare" --> CycleLoss;
    y_hatB2 -- "Compare" --> CycleLoss;
    yB -- "Compare" --> CycleLoss;
    CycleLoss -- "Guides" --> G;
    CycleLoss -- "Guides" --> F;


    %% CLIP Loss Calculation
    y_hatB -- "Image_Input" --> CLIP;
    PromptB -- "Text_Input" --> CLIP;
    x_hatA2 -- "Image_Input" --> CLIP;
    PromptA -- "Text_Input" --> CLIP;
    CLIP -- "Calculates_Similarity" --> CLIPLoss;
    CLIPLoss -- "Guides_G" --> G;
    CLIPLoss -- "Guides_F" --> F;


    %% Identity Loss Calculation
    yB -- "Input_Target_B" --> G -- "Generates" --> y_ident[Identity_B];
    xA -- "Input_Target_A" --> F -- "Generates" --> x_ident[Identity_A];
    y_ident -- "Compare" --> IdentLoss;
    yB -- "Compare" --> IdentLoss;
    x_ident -- "Compare" --> IdentLoss;
    xA -- "Compare" --> IdentLoss;
    IdentLoss -- "Guides_G" --> G;
    IdentLoss -- "Guides_F" --> F;


    %% Styling
    style G fill:#D5E8D4,stroke:#82B366,stroke-width:2px
    style F fill:#D5E8D4,stroke:#82B366,stroke-width:2px
    style CLIP fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px
    style CycleLoss fill:#F8CECC,stroke:#B85450,stroke-width:1px,stroke-dasharray: 5 5
    style CLIPLoss fill:#F8CECC,stroke:#B85450,stroke-width:1px,stroke-dasharray: 5 5
    style IdentLoss fill:#F8CECC,stroke:#B85450,stroke-width:1px,stroke-dasharray: 5 5
    style xA fill:#FFE6CC,stroke:#D79B00
    style yB fill:#FFE6CC,stroke:#D79B00
    style PromptA fill:#E1D5E7,stroke:#9673A6
    style PromptB fill:#E1D5E7,stroke:#9673A6
