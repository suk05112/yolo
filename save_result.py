#model_duration
#dataload_duration

#detect_duration
#pt_duration

def save(duration1, duration2, img):
    with open('/home/isl/.jupyter/custom/dataset/duration.txt', 'a') as f:
      f.write('\n' + "this is test" + '\n' + str(duration1) + " " + str(duration2) )
    print("save duration.txt")