# 5th Place Winning Solution - ISIC 2024 - Skin Cancer Detection with 3D-TBP
This is the repository of part of 5th winning solution of "ISIC 2024 - Skin Cancer Detection with 3D-TBP" in kaggle.
![solution](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3948967%2F38ce6eb11d8320552850efd2a20a1d87%2Fimage1.png?generation=1725877390031631&alt=media)

This repository contains these two models in the pipeline above. 

- Tabular Model1
- Image Model1


## Contents
- kaggle_isic
  - colab
    - Google colab was used to train image models.
    - notebooks in this folder is supposed to run on colab.
  - data
    - where competition data and outputs of the codes are placed.
  - notebooks
    - several temporal files to experiment ensemble with my teammates
  - run
    - conf
      - configuration files written in hydra format
    - running files are placed here. you can run these files by the below command.
    - `python -m run.lgb`
  - src
    - modules imported from running files. 


## Hardware
Mac Studio with Apple M2 Max, 32GB memory.
