from augmentations import *
import utils
import config



DataLoader = data_loader(config.question_train_path, config.answer_train_path)

# For each Image in the Data loader, for each question -- call language only augmentations
# For each image in the dataloader, call image augmentations
# select some of the questions from these questions randomly.
# Add the QA pairs to the dataset by calling utils.

# Qn: Should change all augmentations to class structure or not?