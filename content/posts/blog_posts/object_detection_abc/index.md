---
author: "Bedir Tapkan"
title: "Object Detection ABCs - Setting Up Metrics"
date: 2022-11-22
description: "A little tutorial on setting up object detection metrics and basics, prepping for YOLO implementation."
tags: ["Machine Learning", "Deep Learning", "Computer Vision", "Object Detection"]
ShowToc: true
---

These are my notes on refreshing my object detection knowledge. We will start with bounding boxes for localization and cover everything we need before jumping in to implement YOLO algorithms.

This tutorial includes answers to the following questions:

- What is localization?
- What are a bounding box and sliding window?
- How to measure the success of a predicted bounding box: Intersection over the union.
- How to get rid of extra bounding boxes: Non-max suppression.
- Evaluation Metric for Object Detection: Mean average precision

##### [Check references for addresses of the images.](#references)

## Object Detection

Object Detection is finding objects in an image and where they are located in the image. Adding localization or location on detected objects for a classification task will give us object detection.

We mainly use Deep Learning approaches for modern applications (what a surprise 🙂). On the other hand, object detection focuses on how to solve localization problems for the most part, so we will focus on some methods to help us solve this issue to begin with.

## Localization

Localization is an easy concept. For example, imagine we have an image of a cat; I can classify this image as a cat with some confidence level. If I want to show where the cat is in the image, I need to use localization; to determine what part of the image the cat is at. Similarly, if we had multiple objects on the scene, we could detect each separately, classifying the image as numerous. *We call this location identification of various objects **localization***.

![image from [https://www.datacamp.com/tutorial/object-detection-guide](https://www.datacamp.com/tutorial/object-detection-guide)](images/img.png)



While we try to classify an image, our model will also predict where is the predicted object located, or rather where the bounding box is located. To do this, we have additional parameters to describe the location of the bounding box. For example, we can define a bounding box using the coordinates of its corners or the location of its middle point and height and weight. I'll talk about that later.

## Bounding Box & Sliding Window

As we discussed, a bounding box surrounds an object in the image. The red, green and blue boxes in the image above are examples of bounding boxes. That’s great; how do we handle drawing these boxes, though?

There are many ways proposed, and I am sure the research will continue on it for some time, but the primary approach we have is called a sliding window. A sliding window is to have a box run around all the images and try to find which part of the image actually has the object we are predicting. 

![image from [https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/](https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)](images/img1.png)

As we can guess, this is a slow method, considering how many boxes there are for every image (you also have to run your model on each window). So there is some work on improving this method's speed. 

The next problem is getting multiple bounding boxes for an image. We will see how to handle this as well.

## Intersection over Union

This is a simple method to calculate the error of a given prediction. We check the intersection of the real bounding box and the prediction, and divide it into the union of the two. Very simple, isn’t it? 

![image from [https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)](images/img2.png)

What we need to do now is to write a simple geometric formula to determine the area of intersection and the union. Let’s jump in using PyTorch. For the sake of understanding, I will first give the non-vectorized implementation, then upgrade the lines to vectorized version

The first thing we need to do is to convert the midpoint representation to corners. W

```python
# We are given representations for both boxes: box_pred, box_target
# both are torch.Tensor with dimensions (N, 4)
# Let's assume we also describe the method of representation: box_format \in ['corners', 'midpoint']
if box_format == "midpoint":
    # Convert midpoint to corners
    for i in range(len(pred_box)):
        w, h = pred_box[i, 2], pred_box[i, 3]

        pred_box[i, 0] = pred_box[i, 0] - w / 2
        pred_box[i, 1] = pred_box[i, 1] - h / 2
        pred_box[i, 2] = pred_box[i, 0] + w
        pred_box[i, 3] = pred_box[i, 1] + h

    for i in range(len(target_box)):
        w, h = target_box[i, 2], target_box[i, 3]

        target_box[i, 0] = target_box[i, 0] - w / 2
        target_box[i, 1] = target_box[i, 1] - h / 2
        target_box[i, 2] = target_box[i, 0] + w
        target_box[i, 3] = target_box[i, 1] + h
```

We are basically just iterating through all the boxes we have in the list and changing their values using width and height from the middle point. If we know the middle point, we can remove half of the width and height to find the top left corner of the image (in python images are 0,0 on the top left and getting higher numbers towards south and east). So the formula for the top left corner is $x_1 = m_x - \frac{w}{2}$ where $m_x$ is the x for the middle point and $w$ is the width. The same logic goes for y. If we add $w$ to this value we will find the $x_2$.

Now if we do this way, we are not making use of tensor operations, so let’s alter the code to get a faster calculation

```python
if box_format == "midpoint":
    # Convert midpoint to corners
    pred_box = torch.cat((pred_box[..., :2] - pred_box[..., 2:] / 2, pred_box[..., :2] + pred_box[..., 2:] / 2), dim=1)
    target_box = torch.cat((target_box[..., :2] - target_box[..., 2:] / 2, target_box[..., :2] + target_box[..., 2:] / 2), dim=1)
```

If you didn’t get what’s happening here, please ponder over the code a little to grasp how the two of them are the same.

Now that we setup the corners, we need to find the intersection and the union of the areas. To find the area we can simply multiply the height and width which are equal to the distance between the x’s and y’s. So $A = abs(y_2-y_1)\times abs(x_2-x_1)$. We can get the area for boxes with this logic

```python
for i in range(len(pred_box)):
    box1_x_diff = (pred_box[i, 2] - pred_box[i, 0])
    box1_y_diff = (pred_box[i, 3] - pred_box[i, 1])
    box1_area = box1_x_diff * box1_y_diff

    box2_x_diff = (target_box[i, 2] - target_box[i, 0])
    box2_y_diff = (target_box[i, 3] - target_box[i, 1])
    box2_area = box2_x_diff * box2_y_diff
```

We now have everything but the intersection. To find the intersection we can use a simple idea. 

- The $x$ for the first point (top left corner of the intersection) will be the maximum of the $x_1$ of the target and the prediction.
- The $y$ for the first point will be the maximum of the $y_1$’s.
- In the same way, the $x_2$ will be the minimum of the two $x_2$’s.
- $y_2$ will be the minimum of the current $y_2$’s.

So we can just find the corners and use the same logic as the boxes to find the area of the intersection `intersection = (x2 - x1) * (y2 - y1)`. Though we need a little extra here, we have a probability that there is nothing at the intersection in which case we need to just say so, meaning we need to increase the value to `0`. 

```python
for i in range(len(pred_box)):
    x1 = max(pred_box[i, 0], target_box[i, 0])
    y1 = max(pred_box[i, 1], target_box[i, 1])
    x2 = min(pred_box[i, 2], target_box[i, 2])
    y2 = min(pred_box[i, 3], target_box[i, 3])

    x_diff = x2 - x1 if x2 - x1 > 0 else 0
    y_diff = y2 - y1 if y2 - y1 > 0 else 0
    intersection = x_diff * y_diff
```

We could just use `clamp(0)` instead of an `if-else` statement there, but I wanted to make it as easy to comprehend as possible.

Let’s combine everything and PyTorchify at the same time

```python
def intersection_over_union(
    pred_box: torch.Tensor, target_box: torch.Tensor, box_format: str = "midpoint"
) -> torch.Tensor:
    """
    Calculates intersection over union (IoU) for two sets of boxes
    
    Args:
        pred_box: (tensor) Bounds for the predicted boxes, sized [N,4]
        target_box: (tensor) Bounds for the target boxes, sized [N,4]
        box_format: (str) midpoint/corners, if boxes are (x,y,w,h) or (x1,y1,x2,y2)
    """
    if box_format == "midpoint":
        # Convert midpoint to corners
        pred_box = torch.cat(
            (pred_box[..., :2] - pred_box[..., 2:] / 2, 
             pred_box[..., :2] + pred_box[..., 2:] / 2), dim=1)
        target_box = torch.cat(
            (target_box[..., :2] - target_box[..., 2:] / 2, 
             target_box[..., :2] + target_box[..., 2:] / 2), dim=1)

    # Get the coordinates of bounding boxes
    x1 = torch.max(pred_box[..., 0], target_box[..., 0])
    y1 = torch.max(pred_box[..., 1], target_box[..., 1])
    x2 = torch.min(pred_box[..., 2], target_box[..., 2])
    y2 = torch.min(pred_box[..., 3], target_box[..., 3])

    # Intersection area
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Union Area
    box1_area = (pred_box[..., 2] - pred_box[..., 0]) * (pred_box[..., 3] - pred_box[..., 1])
    box2_area = (target_box[..., 2] - target_box[..., 0]) * (target_box[..., 3] - target_box[..., 1])
    union = box1_area + box2_area - intersection

    return intersection / union  # iou
```

You can check the easy version on the [GitHub repo](none). 

That’s all for the IOU! Now let’s jump over to non-max suppression.

## Non-max Suppression

As we mentioned before we might get multiple bounding boxes that fit an object. We need to clean these up and keep only one (one box to rule them all…). We introduce non-max suppression precisely to do this.

![image from [https://pjreddie.com/darknet/yolov1/](https://pjreddie.com/darknet/yolov1/)](images/img3.png)

For each object in our scene we get multiple boxes around, and we need to see if these boxes are actually for the same object and if so we should remove them and keep a single one.

For this, we get all the boxes that say this part of the image is a dog with some confidence. We pick the box with the most confidence and compare all the others with this box using IoU. After that, by using some threshold value, we remove all the boxes that are above the threshold. 

Before all this, we can also discard all the boxes that are below some confidence level, which would ease our job a little.

One last thing to mention before jumping in the code, we do this separately for each class. So for bikes, we would go over the boxes one more time, and for cars too etc.

Time for the code!

So to begin with, we will assume we got some boxes `bboxes` as a tensor, `iou_threshold` for the IoU comparison and `threshold` for confidence threshold.

We first handle the conversion from `h` and `w` as before.

```python
if box_format == "midpoint":
    for box in bboxes:
        w, h = box[4], box[5]
        box[2] = box[2] - w / 2
        box[3] = box[3] - h / 2
        box[4] = box[2] + w
        box[5] = box[3] + h
```

Now that we have proper variables, we then eliminate the boxes that are below the prediction threshold, then we sort the boxes based on their probabilities (so we can consider the highest probability first.)

```python
bboxes = [box for box in bboxes if box[1] > threshold]
bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
```

Then we simply iterate through each box and remove all the boxes that have a higher IoU value than the threshold we gave (we also keep the boxes from other classes). We then append the box we examined for among the boxes to keep.

```python
while bboxes:
    chosen_box = bboxes.pop(0)
    coords = chosen_box[2:]
    bboxes = [
        box for box in bboxes if box[0] != chosen_box[0] or
        intersection_over_union(
            torch.tensor(coords),
            torch.tensor(box[2:]),
            box_format=box_format) < iou_threshold
    ]
    bboxes_after_nms.append(chosen_box)
```

That’s all, let’s bring it all together.

```python
def non_max_suppression(
    bboxes: list, iou_threshold: float, threshold: float, box_format: str = "corners"
) -> List:
    """
    Does Non Max Suppression given bboxes
    
    Args:
        bboxes: (torch.tensor) All bboxes with their class probabilities,
            shape: [N, 6] (class_pred, prob, x1, y1, x2, y2) or
                   [N, 6] (class_pred, prob, x, y, w, h)
        iou_threshold: (float) threshold where predicted bboxes is correct
        threshold: (float) threshold to remove predicted bboxes
        box_format: (str) "midpoint" or "corners" used to specify bboxes
    """
    assert type(bboxes) == list

    # Converting midpoint to corners
    if box_format == "midpoint":
        for box in bboxes:
            w, h = box[4], box[5]
            box[2] = box[2] - w / 2
            box[3] = box[3] - h / 2
            box[4] = box[2] + w
            box[5] = box[3] + h

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        coords = chosen_box[2:]
        bboxes = [
            box for box in bboxes if box[0] != chosen_box[0] or
            intersection_over_union(
                torch.tensor(coords),
                torch.tensor(box[2:]),
                box_format=box_format) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms
```

Done with that as well, we now can focus on the boxes we actually care about, next up is mean average precision.

## Mean Average Precision

So we have an object detection model, how do we evaluate this? The most common metric out there (currently) is the **Mean Average Precision (mAP).** As we do, we will quickly cover the basics and jump into code.

We trained our model, now we are testing using the test or validation data. We will use **precision/recall** to evaluate. So before doing more, let’s go over precision and recall really quickly.

### Precision/Recall

When we make a prediction, we are either right or wrong. Though we can be wrong in different ways. We can say false to something that was true, or true to something that was false. This introduces the idea of **False Positive** and **False Negative.** False positive is when the predicted value is positive but the actual result is negative, and False negative is vice versa. Of course, for this, we need to define truth values to the results.

Other notions introduced here are **True Positive** and **True Negative.** True positives are the true values our model got to predict right, and true negatives are the negative values where our model got it right.

In our case, for object detection, the predictions we make are the positives, and the predictions we didn’t make are the negatives. So false negatives would be the target boxes that we could not predict (I will explain in a bit how we say if we actually predicted a box right, though you can already guess). If we combine true positives and false positives we get all the predictions we made. If we divide the correct predictions from all predictions we get precision, so $p=\frac{TP}{TP + FP}$. If we combine all the truths, so all the target values whether or not we predicted right, we can reach recall $r = \frac{TP}{TP + FN}$. The diagram below explains it perfectly.

![image from [https://towardsdatascience.com/whats-the-deal-with-accuracy-precision-recall-and-f1-f5d8b4db1021](https://towardsdatascience.com/whats-the-deal-with-accuracy-precision-recall-and-f1-f5d8b4db1021)](images/img4.png)

### Back to mAP

Now that we know what precision and recall are, how are they used for evaluation in our case?

First of all, how do we know if a prediction is wrong? Yes, we will use IoU as described above. If the IoU value (with a target) is greater than some threshold we will assume that box is correct.

Here are the steps for finding mean average precision:

- First, we will find the truth values (TP, FP) of all the predictions we made.
- Then we will sort the boxes based on their confidence score (just as before).
- Then we will iterate through the boxes (starting from the highest confidence) and calculate precision and recall for the values up to that point. For example, if we have ten target boxes, and the first box in our list has a TP as the result; we will have a precision of $1/1$ and recall of $1/10$. Let’s say the second one is an FP, then precision will get to $1/2$ and recall will stay the same. Long story short, whenever we see an FP we will increment only the **denominator on the precision** and if we see a TP we will increment **both nominators.**
- Next, we will calculate the **area under the P/R curve** which will be an average precision for a class.
- Then, we do all these again for all the classes and average the results.
- Lastly, we must use different IoU values to do the same thing and get the average of the results.

Well, that seemed longer than it actually is, let’s dive into code to get a better grasp.

### Code

To make things easier to follow, I want to start with the function definition, so we have all the variables set in place before we piece everything else together.

```python
def mean_average_precision(
    pred_boxes: list, true_boxes: list, iou_threshold: float = 0.5, box_format: str = "midpoint", num_classes: int = 20
) -> float:
		"""
		Args:
        pred_boxes: (list) list of lists containing all bboxes with each bboxes 
            specified as [train_idx, class_pred, prob_score, x1, y1, x2, y2] or
                         [train_idx, class_pred, prob_score, x, y, w, h]
        target_boxes: (list) similar to pred_boxes except all the correct ones
        iou_threshold: (float) threshold where predicted bboxes is correct
        box_format: (str) "midpoint" or "corners" used to specify bboxes
        num_classes: (int) number of classes
    """
```

After that, we will continue with the first step as usual: convert the point format…

In the main part, we will iterate through all the classes, and keep our attention on those only. So to do that we get the targets and predictions for a single class to begin with. We also create a list to keep track of the average precisions.

```python
average_precisions = []

for c in range(num_classes):
    # Getting all the detections with the particular class
    detections = [box for box in pred_boxes if box[1] == c]
    # Getting all the ground truth boxes with the particular class
    ground_truths = [box for box in target_boxes if box[1] == c]
```

We will use only these boxes for our next steps (so we only focus on one class at a time). This is preferred since we need to check each box with possible targets. It will make more sense in a bit.

Next up, we sort our predictions based on their probabilities and create a couple of variables for tracking and all. We define `precisions` list for keeping true positives and false positives. 1’s will be TP and 0’s will be FP.

```python
# Counting the number of bboxes
n_predicted = len(c_predicted)
n_target = len(c_target)

# If there are no predictions or no targets then AP is 0
if n_predicted == 0 or n_target == 0:
    average_precisions.append(0)
    continue

# Sorting the predictions by the probability score
c_predicted = sorted(c_predicted, key=lambda x: x[2], reverse=True)

# Defining a list to keep track of which target bboxes have
# already been matched to a prediction
target_boxes_already_matched = []

# Defining a list to keep track of the precision at each detection
# (i.e. for each prediction)
precisions = []
```

Now that we are set, we will iterate through all the predictions. While only considering the target boxes that are for the same image we will check each target and decide if the prediction we are checking passes the IoU threshold for that target. If so we will mark that target done so we don’t consider it for the next prediction. We also will add a true positive to our precisions (adding 1).

```python
# Iterating through all the predicted bboxes
for prediction in c_predicted:
    # Getting the image index
    img_idx = prediction[0]

    # Getting the target boxes which correspond to the same image
    # as the prediction
    target_boxes_with_same_img_idx = [
        box for box in c_target if box[0] == img_idx
    ]

    # If there are no target boxes in the image then the prediction
    # is automatically a false positive
    if len(target_boxes_with_same_img_idx) == 0:
        precisions.append(0)
        continue

    # Iterating through all the target bboxes in the image
    for target in target_boxes_with_same_img_idx:
        # If the target bbox has already been matched to a prediction 
        # then we skip
        if target in target_boxes_already_matched:
            continue

        # If the IoU between the target and the prediction is above
        # the threshold then the prediction is a true positive
        if intersection_over_union(
            torch.tensor(prediction[3:]),
            torch.tensor(target[3:]),
            box_format=box_format
        ) > iou_threshold:
            target_boxes_already_matched.append(target)
            precisions.append(1)
        else:
            precisions.append(0)
```

Now we need to calculate precisions and recalls, just like we mentioned while explaining the algorithm (adding to the nominator/denominator thing). We will also add an extra zero to make the graph (for AUC) go from 0 to 1. Lastly, we use the trapezoidal rule to calculate the AUC for precision recall.

```python
# If all the predictions are false positives then precision is 0
if sum(precisions) == 0:
    average_precisions.append(0)
    continue

# Calculating the precision and recall at each detection
precisions = [sum(precisions[:i+1]) / (i+1) for i in range(n_predicted)]
recalls = [sum(precisions[:i+1]) / (n_target + epsilon) for i in range(n_predicted)]

# Adding an extra precision and recall value of 0 and 1 respectively
# to make the graph go from 0 to 1
precisions.insert(0, 0)
recalls.insert(0, 0)

# Calculating the average precision using the precision-recall curve 
# using the trapezoidal rule in pytorch
average_precisions.append(torch.trapz(torch.tensor(precisions), torch.tensor(recalls)))
```

Let’s put all the bells and whistles together and get our fully formed function:

```python
def mean_average_precision(
    pred_boxes: list, target_boxes: list, iou_threshold: float = 0.5, 
    box_format: str = "midpoint", num_classes: int = 20
) -> float:
    """
    Calculates mean average precision
    
    Args:
        pred_boxes: (list) list of lists containing all bboxes with each bboxes 
            specified as [img_idx, class_pred, prob_score, x1, y1, x2, y2] or
                         [img_idx, class_pred, prob_score, x, y, w, h]
        target_boxes: (list) similar to pred_boxes except all the correct ones
        iou_threshold: (float) threshold where predicted bboxes is correct
        box_format: (str) "midpoint" or "corners" used to specify bboxes
        num_classes: (int) number of classes
    """
    # Starting by defining a list for all AP for each class
    average_precisions = []

    for class_ in range(num_classes):
        # Getting all the detections with the particular class
        c_predicted = [box for box in pred_boxes if box[1] == class_]
        # Getting all the ground truth boxes with the particular class
        c_target = [box for box in target_boxes if box[1] == class_]

        # Counting the number of bboxes
        n_predicted = len(c_predicted)
        n_target = len(c_target)

        # If there are no predictions or no targets then AP is 0
        if n_predicted == 0 or n_target == 0:
            average_precisions.append(0)
            continue

        # Sorting the predictions by the probability score
        c_predicted = sorted(c_predicted, key=lambda x: x[2], reverse=True)
        
        # Defining a list to keep track of which target bboxes have
        # already been matched to a prediction
        target_boxes_already_matched = []

        # Defining a list to keep track of the precision at each detection
        # (i.e. for each prediction)
        precisions = []

        # Iterating through all the predicted bboxes
        for prediction in c_predicted:
            # Getting the image index
            img_idx = prediction[0]

            # Getting the target boxes which correspond to the same image
            # as the prediction
            target_boxes_with_same_img_idx = [
                box for box in c_target if box[0] == img_idx
            ]

            # If there are no target boxes in the image then the prediction
            # is automatically a false positive
            if len(target_boxes_with_same_img_idx) == 0:
                precisions.append(0)
                continue

            # Iterating through all the target bboxes in the image
            for target in target_boxes_with_same_img_idx:
                # If the target bbox has already been matched to a prediction 
                # then we skip
                if target in target_boxes_already_matched:
                    continue

                # If the IOU between the target and the prediction is above
                # the threshold then the prediction is a true positive
                if intersection_over_union(
                    torch.tensor(prediction[3:]),
                    torch.tensor(target[3:]),
                    box_format=box_format
                ) > iou_threshold:
                    target_boxes_already_matched.append(target)
                    precisions.append(1)
                else:
                    precisions.append(0)

        # If all the predictions are false positives then precision is 0
        if sum(precisions) == 0:
            average_precisions.append(0)
            continue

        # Calculating the precision and recall at each detection
        precisions = [sum(precisions[:i+1]) / (i+1) for i in range(n_predicted)]
        recalls = [sum(precisions[:i+1]) / (n_target + epsilon) 
									 for i in range(n_predicted)]

        # Adding an extra precision and recall value of 0 and 1 respectively
        # to make the graph go from 0 to 1
        precisions.insert(0, 0)
        recalls.insert(0, 0)

        # Calculating the average precision using the precision-recall curve 
        # using the trapezoidal rule in pytorch
        average_precisions.append(
					torch.trapz(torch.tensor(precisions), torch.tensor(recalls)))

    return sum(average_precisions) / len(average_precisions)
```

That’s it! Wait… One last thing. This was for a single IoU threshold, we will need more than that. Let’s write a simple function that calls our `mean_average_precision` .

```python
def map_driver(pred_boxes: list, target_boxes: list, starting_iou=0.5, 
               increment=0.05, ending_iou=0.9, num_classes=20, 
               box_format="midpoint") -> float:
    """
    Calculates the mean average precision for a range of IOU thresholds
    
    Args:
        pred_boxes: (list) list of lists containing all bboxes with each bboxes
            specified as [img_idx, class_pred, prob_score, x1, y1, x2, y2] or
                         [img_idx, class_pred, prob_score, x, y, w, h]
        target_boxes: (list) same as the bbox list but contains the correct
            bboxes
        starting_iou: (float) starting IOU threshold
        increment: (float) increment to increase the IOU threshold by
        ending_iou: (float) ending IOU threshold
        num_classes: (int) number of classes
        box_format: (str) "midpoint" or "corners" used to specify bboxes
    """
    mean_average_precisions = []

    for iou_threshold in np.arange(starting_iou, ending_iou, increment):
        mean_average_precisions.append(
            mean_average_precision(
                pred_boxes, target_boxes, iou_threshold, box_format, num_classes
            )
        )

    return mean_average_precisions
```

And we are fully done. Now we know every bit we need to actually go ahead and implement our first object detection algorithm, which will be the first version of the still state-of-the-art YOLO algorithm. It now has YOLOv7, but we will start with implementing v1.

You can find all the code from this tutorial [here](https://github.com/BedirT/DeepLearningPractices/blob/main/object_detection/tools.py).

## References

- [https://www.youtube.com/c/AladdinPersson](https://www.youtube.com/c/AladdinPersson)
- [https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [https://labelyourdata.com/articles/mean-average-precision-map](https://labelyourdata.com/articles/mean-average-precision-map)
- [https://en.wikipedia.org/wiki/Precision_and_recallhttps://web.archive.org/web/20191114213255/https://www.flinders.edu.au/science_engineering/fms/School-CSEM/publications/tech_reps-research_artfcts/TRRA_2007.pdf](https://web.archive.org/web/20191114213255/https://www.flinders.edu.au/science_engineering/fms/School-CSEM/publications/tech_reps-research_artfcts/TRRA_2007.pdf)
- Image 1 from [https://www.datacamp.com/tutorial/object-detection-guide](https://www.datacamp.com/tutorial/object-detection-guide)
- Image 2 from [https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/](https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)
- Image 3 from [https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
- Image 4 from [https://pjreddie.com/darknet/yolov1/](https://pjreddie.com/darknet/yolov1/)
- Image 5 from [https://towardsdatascience.com/whats-the-deal-with-accuracy-precision-recall-and-f1-f5d8b4db1021](https://towardsdatascience.com/whats-the-deal-with-accuracy-precision-recall-and-f1-f5d8b4db1021)