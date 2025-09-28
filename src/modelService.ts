import * as tf from '@tensorflow/tfjs';

// Define the shape of our training data and prediction data.
// This is important for type safety in a TypeScript project.
interface TrainingData {
  features: tf.Tensor2D;
  labels: tf.Tensor1D;
}

/**
 * Creates and compiles a simple neural network model for binary classification.
 * @param learningRate The learning rate for the Adam optimizer.
 */
export function createModel(learningRate: number): tf.Sequential {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [30], units: 20, activation: 'relu' }),
      tf.layers.dense({ units: 10, activation: 'relu' }),
      tf.layers.dense({ units: 2, activation: 'softmax' }), // Output layer for 2 classes (Malignant/Benign)
    ],
  });

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

/**
 * Generates synthetic data for training and validation purposes.
 * This function mimics the behavior of the original `app.js` data generation.
 * @param numSamples The number of samples to generate.
 * @returns A TrainingData object containing features and labels.
 */
export function generateSyntheticData(numSamples: number): TrainingData {
  // Generate random data for 30 features
  const features = tf.randomNormal<tf.Rank.R2>([numSamples, 30]);
  // Generate random labels (0 or 1), now explicitly cast to float32
  const labels = tf.randomUniform<tf.Rank.R1>([numSamples], 0, 2, 'int32').toFloat() as tf.Tensor1D;
  return { features, labels };
}

/**
 * Simulates the training process of the neural network.
 * @param model The TensorFlow.js model to train.
 * @param trainingData The data used for training.
 * @param epochs The number of training epochs.
 * @param batchSize The size of the mini-batches for training.
 * @param onEpochEnd A callback function to report training progress.
 */
export async function trainModel(
  model: tf.Sequential,
  trainingData: TrainingData,
  epochs: number,
  batchSize: number,
  onEpochEnd: (epoch: number, loss: number, accuracy: number) => void
): Promise<void> {
  const { features, labels } = trainingData;

  await model.fit(features, labels, {
    epochs,
    batchSize,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const loss = logs?.loss ?? 0;
        const accuracy = logs?.acc ?? 0;
        onEpochEnd(epoch + 1, loss, accuracy);
      },
    },
  });
}

/**
 * Normalizes input data for prediction.
 * This is a simple standardization based on mean and standard deviation.
 * @param inputData The raw input data array.
 * @returns An array of normalized numbers.
 */
export function normalizeData(inputData: number[]): number[] {
  const means = [14.13, 19.29, 91.97, 654.89, 0.096, 0.104];
  const stds = [3.52, 4.30, 24.30, 351.91, 0.014, 0.053];

  const normalizedData = inputData.map((val, idx) => {
    // Only normalize the first 6 features for simplicity
    if (idx < means.length) {
      return (val - means[idx]) / stds[idx];
    }
    return val;
  });

  return normalizedData;
}