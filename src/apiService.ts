// Define the expected structure of the prediction response from the backend.
export interface PredictionResult {
  diagnosis: string;
  confidence: number;
  accuracy: number;
  prediction_timestamp?: string;
  probability_benign?: number;
  probability_malignant?: number;
}

// Batch prediction result
export interface BatchPredictionResult {
  predictions: Array<{
    row_index: number;
    diagnosis: string;
    confidence: number;
    probability_benign: number;
    probability_malignant: number;
  }>;
  total_predictions: number;
  timestamp: string;
}

// Define file upload response structure
export interface FileUploadResult {
  success: boolean;
  message: string;
  filename?: string;
  columns?: string[];
  sample_data?: any[];
  total_rows?: number;
}

// Define error response structure
interface ApiError {
  error: string;
  details?: any;
}

// Define dataset response structure
export interface DatasetResponse {
  data: Array<Record<string, any>>;
  total_count: number;
  feature_names: string[];
  timestamp: string;
}

// Define model status response structure
export interface ModelStatusResponse {
  trained: boolean;
  accuracy: number;
  total_samples: number;
  feature_count: number;
  feature_names?: string[];
  timestamp: string;
}

// Define training response structure
export interface TrainingResponse {
  status: string;
  new_accuracy: number;
  samples_used: number;
  timestamp: string;
  metrics?: any;
}

// Define health check response structure
export interface HealthCheckResponse {
  status: string;
  model_loaded: boolean;
  data_loaded: boolean;
  feature_count: number;
  timestamp: string;
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
 * Generic API call function for multipart/form-data requests (file uploads)
 */
async function apiCallMultipart<T>(endpoint: string, formData: FormData): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errorMessage = `API call failed with status: ${response.status}`;
      
      try {
        const errorData: ApiError = await response.json();
        errorMessage = errorData.error || errorMessage;
      } catch {
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
 * Batch predict from CSV file upload (no diagnosis column needed)
 * @param file CSV file with patient data
 * @returns Batch prediction results
 */
export async function batchPredictFromFile(file: File): Promise<BatchPredictionResult> {
  if (!file) {
    throw new Error('No file provided');
  }

  if (!file.type.includes('csv') && !file.name.endsWith('.csv')) {
    throw new Error('File must be a CSV file');
  }

  const formData = new FormData();
  formData.append('file', file);
  
  return apiCallMultipart<BatchPredictionResult>('/predict/batch', formData);
}

/**
 * Get a random sample from the current dataset
 * @returns Random sample data with features
 */
export async function getRandomSample(): Promise<{
  features: number[];
  diagnosis?: string;
  feature_names: string[];
}> {
  return apiCall('/data/random-sample');
}

/**
 * Fetches the current model status and statistics.
 * @returns A promise that resolves to model status information.
 */
export async function getModelStatus(): Promise<ModelStatusResponse> {
  return apiCall<ModelStatusResponse>('/status');
}

/**
 * Fetches dataset with optional filtering.
 * @param diagnosis Optional filter by diagnosis ('M' or 'B')
 * @param limit Optional limit on number of records (0 or undefined = all records)
 * @returns A promise that resolves to the dataset
 */
export async function getDataset(diagnosis?: string, limit?: number): Promise<DatasetResponse> {
  const params = new URLSearchParams();
  if (diagnosis) params.append('diagnosis', diagnosis);
  if (limit !== undefined && limit !== 0) params.append('limit', limit.toString());
  
  const endpoint = `/data${params.toString() ? `?${params}` : ''}`;
  return apiCall<DatasetResponse>(endpoint);
}

/**
 * Triggers model retraining using row indices.
 * @param selectedIndices Optional array of row indices (0-based) to train on
 * @returns A promise that resolves to retraining results
 */
export async function retrainModel(selectedIndices?: number[]): Promise<TrainingResponse> {
  return apiCall<TrainingResponse>('/train', {
    method: 'POST',
    body: JSON.stringify({ indices: selectedIndices || [] }),
  });
}

/**
 * Uploads a CSV file for training data.
 * @param file The CSV file to upload
 * @returns A promise that resolves to upload result
 */
export async function uploadTrainingFile(file: File): Promise<FileUploadResult> {
  if (!file) {
    throw new Error('No file provided');
  }

  if (!file.type.includes('csv') && !file.name.endsWith('.csv')) {
    throw new Error('File must be a CSV file');
  }

  const maxSize = 10 * 1024 * 1024; // 10MB
  if (file.size > maxSize) {
    throw new Error('File size must be less than 10MB');
  }

  const formData = new FormData();
  formData.append('file', file);
  
  return apiCallMultipart<FileUploadResult>('/upload', formData);
}

/**
 * Gets the list of available uploaded files.
 * @returns A promise that resolves to list of uploaded files
 */
export async function getUploadedFiles(): Promise<{
  files: Array<{
    filename: string;
    upload_time: string;
    size: number;
    columns: string[];
  }>;
}> {
  return apiCall('/files');
}

/**
 * Deletes an uploaded file.
 * @param filename The name of the file to delete
 * @returns A promise that resolves to deletion result
 */
export async function deleteUploadedFile(filename: string): Promise<{
  success: boolean;
  message: string;
}> {
  return apiCall(`/files/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
  });
}

/**
 * Gets detailed information about an uploaded file including column analysis.
 * @param filename The name of the file to analyze
 * @returns A promise that resolves to file analysis
 */
export async function analyzeUploadedFile(filename: string): Promise<{
  filename: string;
  columns: string[];
  total_rows: number;
  sample_data: any[];
  column_types: Record<string, string>;
  missing_values: Record<string, number>;
}> {
  return apiCall(`/files/${encodeURIComponent(filename)}/analyze`);
}

/**
 * Performs a health check on the API.
 * @returns A promise that resolves to health status
 */
export async function healthCheck(): Promise<HealthCheckResponse> {
  return apiCall<HealthCheckResponse>('/health');
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

/**
 * Validates a CSV file on the client side before upload.
 * @param file The file to validate
 * @returns A promise that resolves to validation result
 */
export async function validateCsvFile(file: File): Promise<{
  valid: boolean;
  errors: string[];
  warnings: string[];
  preview?: {
    headers: string[];
    rows: string[][];
    totalRows: number;
  };
}> {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!file.type.includes('csv') && !file.name.endsWith('.csv')) {
    errors.push('File must be a CSV file');
  }

  if (file.size > 10 * 1024 * 1024) {
    errors.push('File size must be less than 10MB');
  }

  if (file.size === 0) {
    errors.push('File cannot be empty');
  }

  if (errors.length > 0) {
    return { valid: false, errors, warnings };
  }

  try {
    const text = await file.text();
    const lines = text.split('\n').filter(line => line.trim());
    
    if (lines.length < 2) {
      errors.push('CSV file must contain at least a header row and one data row');
      return { valid: false, errors, warnings };
    }

    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
    
    const emptyHeaders = headers.filter((h, i) => !h && `Column_${i + 1}`);
    if (emptyHeaders.length > 0) {
      warnings.push('Some columns have empty headers');
    }

    const duplicateHeaders = headers.filter((h, i) => headers.indexOf(h) !== i);
    if (duplicateHeaders.length > 0) {
      warnings.push('Duplicate column headers detected');
    }

    const previewRows = lines.slice(1, 6).map(line => 
      line.split(',').map(cell => cell.trim().replace(/"/g, ''))
    );

    const preview = {
      headers,
      rows: previewRows,
      totalRows: lines.length - 1
    };

    return {
      valid: errors.length === 0,
      errors,
      warnings,
      preview
    };

  } catch (error) {
    errors.push('Failed to parse CSV file. Please check the file format.');
    return { valid: false, errors, warnings };
  }
}