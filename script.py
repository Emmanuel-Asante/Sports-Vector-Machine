# Import modules
import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

# Print out column names
#print(aaron_judge.columns)

# Print out the values for the description feature from aaron_judge dataframe
#print(aaron_judge.description.unique())

# Print out the values for the type feature
#print(aaron_judge.type.unique())

# Create a function to find strike zone
def strike_zone(data_set):

  # Change the values for the type feature
  data_set['type'] = data_set['type'].map({
    'S': 1,
    'B': 0
  })

  # Print the type column from the data_set
  #print(data_set['type'])

  # Print out the plate_x feature from data_set
  #print(data_set['plate_x'])

  # Print out the plate_z feature from data_set
  #print(data_set['plate_z'])

  # Drop NaN values
  data_set = data_set.dropna(subset = ['type', 'plate_x', 'plate_z'])

  # Create subplots
  fig, ax = plt.subplots()

  # create a scatter plot
  plt.scatter(x = data_set.plate_x, y = data_set.plate_z, c = data_set.type, cmap = plt.cm.coolwarm, alpha = 0.25)

  # Split data into a training set and a validation set
  training_set, validation_set = train_test_split(data_set, random_state = 1)

  # Create an SVC object
  classifier = SVC(kernel = 'rbf', gamma = 3, C = 0.9)

  # Train the model
  classifier.fit(training_set[['plate_x', 'plate_z']], training_set.type)

  # Call draw_boundary function
  draw_boundary(ax, classifier)

  # Set axes size
  ax.set_ylim(-2, 6)

  # Set axes size
  ax.set_xlim(-3, 3)

  # Print out the model's score
  print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set.type))

  # Show plot
  plt.show()

# A player's strike zone
strike_zone(aaron_judge)

# A player's strike zone
strike_zone(jose_altuve)

# A player's strike zone
strike_zone(david_ortiz)