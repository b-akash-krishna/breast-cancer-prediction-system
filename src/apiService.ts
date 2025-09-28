// Define the expected structure of the prediction response from the backend.
export interface PredictionResult {
  diagnosis: string;
  confidence: number;
  accuracy: number;
  prediction_timestamp?: string;
}

// Define error response structure
interface ApiError {
  error: string;
  details?: any;
}

/**
 * Base API configuration
 */
const API_BASE_URL = 'http://127.0.0.1:5000';

/**
 * Generic API call function with error handling
 */
async function apiCall<T>(endpoint: string, options?: RequestInit): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      let errorMessage = `API call failed with status: ${response.status}`;
      
      try {
        const errorData: ApiError = await response.json();
        errorMessage = errorData.error || errorMessage;
      } catch {
        // If we can't parse the error response, use the status message
        errorMessage = `${response.status}: ${response.statusText}`;
      }
      
      throw new Error(errorMessage);
    }

    const result: T = await response.json();
    return result;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Unknown API error occurred');
  }
}

/**
 * Sends patient features to the backend API for a breast cancer prediction.
 * @param features An array of numerical features for a patient.
 * @returns A promise that resolves to a PredictionResult object.
 */
export async function predictWithApi(features: number[]): Promise<PredictionResult> {
  if (!Array.isArray(features) || features.length === 0) {
    throw new Error('Features must be a non-empty array of numbers');
  }

  // Validate that all features are numbers
  const invalidFeatures = features.some(f => typeof f !== 'number' || isNaN(f));
  if (invalidFeatures) {
    throw new Error('All features must be valid numbers');
  }

  return apiCall<PredictionResult>('/predict', {
    method: 'POST',
    body: JSON.stringify({ features }),
  });
}

/**
 * Fetches the current model status and statistics.
 * @returns A promise that resolves to model status information.
 */
export async function getModelStatus(): Promise<{
  trained: boolean;
  accuracy: number;
  total_samples: number;
  feature_count: number;
  feature_names?: string[];
}> {
  return apiCall('/status');
}

/**
 * Fetches dataset with optional filtering.
 * @param diagnosis Optional filter by diagnosis ('M' or 'B')
 * @param limit Optional limit on number of records
 * @returns A promise that resolves to the dataset
 */
export async function getDataset(diagnosis?: string, limit?: number): Promise<{
  data: Array<Record<string, any>>;
  total_count: number;
  feature_names: string[];
}> {
  const params = new URLSearchParams();
  if (diagnosis) params.append('diagnosis', diagnosis);
  if (limit && limit > 0) params.append('limit', limit.toString());
  
  const endpoint = `/data${params.toString() ? `?${params}` : ''}`;
  return apiCall(endpoint);
}

/**
 * Triggers model retraining.
 * @param selectedIds Optional array of sample IDs to train on
 * @returns A promise that resolves to retraining results
 */
export async function retrainModel(selectedIds?: string[]): Promise<{
  status: string;
  new_accuracy: number;
  samples_used: number;
}> {
  return apiCall('/train', {
    method: 'POST',
    body: JSON.stringify({ ids: selectedIds || [] }),
  });
}

/**
 * Performs a health check on the API.
 * @returns A promise that resolves to health status
 */
export async function healthCheck(): Promise<{
  status: string;
  model_loaded: boolean;
  data_loaded: boolean;
  feature_count: number;
}> {
  return apiCall('/health');
}

/**
 * Utility function to check if the API is available
 * @returns A promise that resolves to true if API is available, false otherwise
 */
export async function isApiAvailable(): Promise<boolean> {
  try {
    await healthCheck();
    return true;
  } catch {
    return false;
  }
}