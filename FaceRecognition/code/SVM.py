from sklearn import svm
import ImageIO

def my_svm():
    trainSet_x, trainSet_y, testSet_x, testSet_y = ImageIO.loadSVMImage()

    # clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03)
    clf = svm.LinearSVC()

    print('start fit')
    clf.fit(trainSet_x, trainSet_y)
    print('finish fit')

    predictions = [int(a) for a in clf.predict(testSet_x)]
    predictions_2 = [int(a) for a in clf.predict(trainSet_x)]
    num_correct = sum(int(a == y) for a, y in zip(predictions, testSet_y))
    num_correct_2 = sum(int(a == y) for a, y in zip(predictions_2, trainSet_y))

    print ("%s of %s test values correct." % (num_correct, len(testSet_y)))
    print ("%s of %s train values correct." % (num_correct_2, len(trainSet_y)))

if __name__ == "__main__":
    my_svm()