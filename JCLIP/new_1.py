import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse
# from sklearn.linear_model import LogisticRegression
import numpy as np

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='A')
parser.add_argument('--is-local', action='store_true')

args = parser.parse_args()

model, preprocess = clip.load("ViT-B-32.pkl")
classes = open('Dataset/classname.txt').read().splitlines()

# remove the prefix Animal, Thu-dog, Caltech-101, Food-101

new_classes = []
for c in classes:
    c = c.split(' ')[0]
    if c.startswith('Animal'):
        c = c[7:]
    if c.startswith('Thu-dog'):
        c = c[8:]
    if c.startswith('Caltech-101'):
        c = c[12:]
    if c.startswith('Food-101'):
        c = c[9:]
    c = 'a photo of ' + c
    new_classes.append(c)

text = clip.tokenize(new_classes)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)

# training data loading
imgs_dir = 'Dataset/'
train_labels = open('Dataset/train.txt').read().splitlines()
import random

random.shuffle(train_labels)

train_imgs = [l.split(' ')[0] for l in train_labels]
train_labels = [jt.float32([int(l.split(' ')[1])]) for l in train_labels]

# 每个类挑400张图，根据train_labels中的label来挑选
cnt = {}
new_train_imgs = []
new_train_labels = []
for i in range(len(train_imgs)):
    label = int(train_labels[i].numpy())
    if label not in cnt:
        cnt[label] = 0
    if cnt[label] < 350:
        new_train_imgs.append(train_imgs[i])
        # [1,]
        new_train_labels.append(train_labels[i])
        # print("training labels: ", train_labels[i].shape)
        cnt[label] += 1

# calculate image features of training data
train_features = []
print('Training data processing:')
with jt.no_grad():
    for img in tqdm(new_train_imgs):
        try:
            img = os.path.join(imgs_dir, img)
            image = Image.open(img)
            image = preprocess(image).unsqueeze(0)
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # print("image features: ", image_features.shape)
            # exit(0)
            # [1, 512,]
            train_features.append(image_features)
        except:
            continue

train_features = jt.cat(train_features)
train_labels = jt.cat(new_train_labels)

train_labels = jt.int32(train_labels)

# training
# classifier = LogisticRegression(random_state=0,
#                                 C=8.960,
#                                 max_iter=10000,
#                                 verbose=1)
# classifier.fit(train_features, train_labels)

num_classes = 374

import model 

classifier = model.Classifier(num_classes=num_classes) if not args.is_local else jt.load('classifier.pkl')

optimizer = jt.optim.Adam(classifier.parameters(), lr=0.0001)

criterion = jt.nn.CrossEntropyLoss()

for epoch in range(100000):
    features = train_features
    # print("features: ", features.shape)
    # [N, 512,]
    logits = classifier(features)
    # print("logits: ", logits.shape)
    # [N, 374,]
    # target = jt.nn.one_hot(train_labels, num_classes)
    target = train_labels
    # print("target: ", target.shape)
    # [N, 374,]
    loss: jt.Var = criterion(logits, target)
    if (epoch + 1) % 1000 == 0:
          print('Epoch: ', epoch + 1, 
                ", celoss: ", loss.mean().item(),
                ", mseloss: ", jt.nn.mse_loss(jt.nn.softmax(logits), jt.nn.one_hot(target, num_classes)).mean().item())
    elif (epoch + 1) % 100 == 0:
        print('Epoch: ', epoch + 1, 
            ", celoss: ", loss.mean().item())
    # exit(0)
    loss.sync()
    # optimizer.backward(loss)
    optimizer.step(loss)
    jt.clean()


# testing dataset loading
split = 'TestSet' + args.split
imgs_dir = 'Dataset/' + split
test_imgs = os.listdir(imgs_dir)

# testing data processing
print('Testing data processing:')
test_features = []
with jt.no_grad():
    for img in tqdm(test_imgs):
        img_path = os.path.join(imgs_dir, img)
        image = Image.open(img_path)
        image = preprocess(image).unsqueeze(0)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        test_features.append(image_features)

test_features = jt.cat(test_features)

# testing
with open('result.txt', 'w') as save_file:
    i = 0
    predictions = classifier(test_features)
    print('Testing: ', predictions)
    for prediction in predictions.tolist():
        prediction = np.asarray(prediction)
        top5_idx = prediction.argsort()[-1:-6:-1]
        save_file.write(test_imgs[i] + ' ' +
                        ' '.join(str(idx) for idx in top5_idx) + '\n')
        i += 1

jt.save(classifier, 'classifier.pkl')
