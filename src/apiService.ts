// Define the expected structure of the prediction response from the backend.
export interface PredictionResult {
  diagnosis: string;
  confidence: number;
  accuracy: number;
}

/**
 * Sends patient features to the backend API for a breast cancer prediction.
 * @param features An array of 30 numerical features for a patient.
 * @returns A promise that resolves to a PredictionResult object.
 */
export async function predictWithApi(features: number[]): Promise<PredictionResult> {
  const response = await fetch('http://127.0.0.1:5000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ features }),
  });

  if (!response.ok) {
    throw new Error(`API call failed with status: ${response.status}`);
  }

  const result: PredictionResult = await response.json();
  return result;
}