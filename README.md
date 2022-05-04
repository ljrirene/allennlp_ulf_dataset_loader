# Allennlp_ULF_Dataset_Loader

## Description
This is an Allen NLP dataloader for the ULF dataset

## Environment
To set the dependencies, run
> pip install -r requirements.txt

## How to run
>allennlp train --dry-run --include-package tagging -s <where you want to store the output> configs/test_reader.jsonnet

## Configuration
To load ulf-stog, just change the configuration into:  
  train_data_path: './ulf-data-1.0/ulf-stog/train/all.json',  
  validation_data_path: './ulf-data-1.0/ulf-stog/test/all.json',  
  
## Output
Checking out.log:
> loading instances: 0it [00:00, ?it/s]2022-05-04 04:16:52,915 - INFO - root - Reading entries from lines in file at ./ulf-data-1.0/ulf/train/all.json. 
> loading instances: 1378it [00:00, 5398.95it/s]. 
> loading instances: 0it [00:00, ?it/s]2022-05-04 04:16:53,167 - INFO - root - Reading entries from lines in file at ./ulf-data-1.0/ulf/test/all.json  
> loading instances: 180it [00:00, 8395.51it/s]. 
> 2022-05-04 04:16:53,188 - INFO - allennlp.data.vocabulary - Fitting token dictionary from dataset.  
> building vocab: 1558it [00:00, 18830.54it/s]. 

  
  The KeyError: 'type' is because there is no defined model yet, I just use {} in the config file.
