import config
import pandas as pd

train_list = []
val_list = []
test_list = []

partition=pd.read_csv(config.PARTITION_FILE)
attribute=pd.read_csv(config.ATTRIBUTE_FILE)

train=partition[partition["partition"]==0].image_id.to_list()
val=partition[partition["partition"]==1].image_id.to_list()
test=partition[partition["partition"]==2].image_id.to_list()
print(train)

attribute=attribute.replace(-1, 0)
attribute.index=attribute.image_id
print(attribute.head())


attribute_train=attribute.loc[train]
attribute_val=attribute.loc[val]
attribute_test=attribute.loc[test]

print(attribute_train.columns)
print(len(attribute_train.columns))


attribute_train.to_csv(config.TRAIN_ATTRIBUTE_LIST, index=False)
attribute_val.to_csv(config.VAL_ATTRIBUTE_LIST, index=False)
attribute_test.to_csv(config.TEST_ATTRIBUTE_LIST, index=False)


