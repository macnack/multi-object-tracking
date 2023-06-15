if __name__ == "__main__":
    score = 0
    all_score = 0
    with open("out.txt", 'r') as result_file:
        with open("bboxes_gt.txt", 'r') as ground_truth_file:
            for result, ground in zip(result_file, ground_truth_file):
                result_filename = result.rsplit()
                ground_filename = ground.rsplit()
                result_number_of_objects = result_file.readline().rsplit()[0]
                ground_number_of_objects = ground_truth_file.readline().rsplit()[0]
                all_score += int(result_number_of_objects)
                if result_filename[0] == ground_filename[0]:
                    if result_number_of_objects == ground_number_of_objects:
                        for i in range(int(result_number_of_objects)):
                            x = result_file.readline().rsplit()[0]
                            y = ground_truth_file.readline().rsplit()[0]
                            if x == y:
                                score+=1
                    else:
                        print("wrong number of objects")
                else:
                    print("wrong filename")

    print("Score: ", score, " from all", all_score)
    print("Accuracy: ", score/float(all_score))
                
