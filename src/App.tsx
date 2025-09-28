import { useState, useEffect } from 'react';
import {
  Brain,
  BarChart2,
  Activity,
  Database,
  Grid,
  AlertCircle,
  CheckCircle,
  Loader,
} from 'lucide-react';
import { predictWithApi, PredictionResult } from './apiService';

// Define the expected structure of a data row from the backend
interface DataRow {
  [key: string]: number | string | null;
}

// Define the expected structure of the model status response
interface ModelStatus {
  accuracy: number;
  trained: boolean;
  total_samples: number;
  feature_count: number;
  feature_names?: string[];
  error?: string;
}

// Define API response structure for data endpoint
interface DataResponse {
  data: DataRow[];
  total_count: number;
  feature_names: string[];
}

// Error Boundary Component
const ErrorBoundary: React.FC<{ children: React.ReactNode; fallback?: React.ReactNode }> = ({ 
  children, 
  fallback = <div className="text-red-500 p-4">Something went wrong. Please refresh the page.</div> 
}) => {
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    const handleError = () => setHasError(true);
    window.addEventListener('error', handleError);
    return () => window.removeEventListener('error', handleError);
  }, []);

  if (hasError) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
};

// Loading Spinner Component
const LoadingSpinner: React.FC<{ message?: string }> = ({ message = "Loading..." }) => (
  <div className="flex flex-col items-center justify-center p-8">
    <Loader className="w-8 h-8 animate-spin text-indigo-600 mb-2" />
    <p className="text-gray-600 dark:text-gray-400">{message}</p>
  </div>
);

// Error Alert Component
const ErrorAlert: React.FC<{ message: string; onRetry?: () => void }> = ({ message, onRetry }) => (
  <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-4">
    <div className="flex items-center">
      <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
      <p className="text-red-700 dark:text-red-300">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="ml-auto text-red-600 hover:text-red-800 underline"
        >
          Retry
        </button>
      )}
    </div>
  </div>
);

// Success Alert Component
const SuccessAlert: React.FC<{ message: string }> = ({ message }) => (
  <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4 mb-4">
    <div className="flex items-center">
      <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
      <p className="text-green-700 dark:text-green-300">{message}</p>
    </div>
  </div>
);

// --- Dashboard Tab Component
const DashboardTab = () => {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchModelStatus = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('http://127.0.0.1:5000/status');
      if (!response.ok) {
        throw new Error(`Failed to fetch model status: ${response.status}`);
      }
      
      const status: ModelStatus = await response.json();
      setModelStatus(status);
    } catch (error) {
      console.error("Failed to fetch dashboard data:", error);
      setError(error instanceof Error ? error.message : 'Failed to fetch dashboard data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModelStatus();
  }, []);

  if (loading) {
    return <LoadingSpinner message="Loading dashboard..." />;
  }

  if (error) {
    return <ErrorAlert message={error} onRetry={fetchModelStatus} />;
  }

  const accuracy = modelStatus?.accuracy ? `${modelStatus.accuracy.toFixed(1)}%` : '--';
  const totalSamples = modelStatus?.total_samples || 0;
  const featureCount = modelStatus?.feature_count || 0;
  const isModelTrained = modelStatus?.trained || false;

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Breast Cancer Classification Dashboard
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          AI-powered diagnostic assistance for medical professionals
        </p>
        {!isModelTrained && (
          <ErrorAlert message="Model is not properly trained. Please check the backend logs." />
        )}
      </div>
      
      <div className="stats-grid grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 flex items-center gap-4">
          <div className={`w-12 h-12 flex items-center justify-center rounded-full text-white ${
            isModelTrained ? 'bg-indigo-500' : 'bg-gray-400'
          }`}>
            <BarChart2 />
          </div>
          <div>
            <div className="text-2xl font-bold">{accuracy}</div>
            <div className="text-sm text-gray-500">Model Accuracy</div>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 flex items-center gap-4">
          <div className="w-12 h-12 flex items-center justify-center rounded-full bg-green-500 text-white">
            <Activity />
          </div>
          <div>
            <div className="text-2xl font-bold">--</div>
            <div className="text-sm text-gray-500">Predictions Made</div>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 flex items-center gap-4">
          <div className="w-12 h-12 flex items-center justify-center rounded-full bg-yellow-500 text-white">
            <Database />
          </div>
          <div>
            <div className="text-2xl font-bold">{totalSamples}</div>
            <div className="text-sm text-gray-500">Training Samples</div>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 flex items-center gap-4">
          <div className="w-12 h-12 flex items-center justify-center rounded-full bg-purple-500 text-white">
            <Grid />
          </div>
          <div>
            <div className="text-2xl font-bold">{featureCount}</div>
            <div className="text-sm text-gray-500">Features</div>
          </div>
        </div>
      </div>

      {modelStatus?.feature_names && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Model Features ({modelStatus.feature_names.length})
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 text-sm">
            {modelStatus.feature_names.map((feature, index) => (
              <div key={index} className="bg-gray-50 dark:bg-gray-700 p-2 rounded">
                {feature}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// --- Train Model Tab Component
const TrainModelTab = () => {
  const [isRetraining, setIsRetraining] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRetrain = async () => {
    setIsRetraining(true);
    setMessage(null);
    setError(null);
    
    try {
      const response = await fetch('http://127.0.0.1:5000/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        throw new Error(`Training failed: ${response.status}`);
      }

      const result = await response.json();
      setMessage(`${result.status}. New accuracy: ${result.new_accuracy?.toFixed(1)}%`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Training failed');
    } finally {
      setIsRetraining(false);
    }
  };

  return (
    <div className="p-8 bg-white dark:bg-gray-800 rounded-lg shadow-xl">
      <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
        Model Training
      </h2>
      
      {error && <ErrorAlert message={error} />}
      {message && <SuccessAlert message={message} />}
      
      <div className="space-y-4">
        <p className="text-gray-600 dark:text-gray-400">
          The model is automatically trained when the Flask backend starts. 
          You can manually retrain it using the current dataset.
        </p>
        
        <button
          onClick={handleRetrain}
          disabled={isRetraining}
          className="flex items-center gap-2 px-6 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 text-white rounded-lg font-medium transition-colors"
        >
          {isRetraining ? (
            <>
              <Loader className="w-4 h-4 animate-spin" />
              Retraining Model...
            </>
          ) : (
            <>
              <Brain className="w-4 h-4" />
              Retrain Model
            </>
          )}
        </button>
      </div>
    </div>
  );
};

// --- Diagnosis Tab Component
const DiagnosisTab = () => {
  const [featureNames, setFeatureNames] = useState<string[]>([]);
  const [inputValues, setInputValues] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch feature names on component mount
  useEffect(() => {
    const fetchFeatureNames = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/status');
        if (response.ok) {
          const status = await response.json();
          if (status.feature_names) {
            setFeatureNames(status.feature_names);
            // Initialize input values
            const initialValues: Record<string, string> = {};
            status.feature_names.forEach((name: string) => {
              initialValues[name] = '';
            });
            setInputValues(initialValues);
          }
        }
      } catch (error) {
        console.error('Failed to fetch feature names:', error);
      }
    };
    
    fetchFeatureNames();
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setInputValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleLoadSampleData = () => {
    // Sample data for first 6 features (most common ones)
    const sampleData: Record<string, string> = {};
    featureNames.forEach((name, index) => {
      if (index < 6) {
        const sampleValues = ['11.76', '21.6', '74.72', '427.9', '0.08637', '0.04966'];
        sampleData[name] = sampleValues[index] || '0';
      } else {
        sampleData[name] = '0';
      }
    });
    
    setInputValues(sampleData);
    setPrediction(null);
    setError(null);
  };

  const handlePredict = async () => {
    setIsLoading(true);
    setPrediction(null);
    setError(null);
    
    try {
      const features = featureNames.map(name => {
        const value = parseFloat(inputValues[name] || '0');
        return isNaN(value) ? 0 : value;
      });

      const result = await predictWithApi(features);
      setPrediction(result);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Prediction failed';
      setError(errorMessage);
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const isFormValid = featureNames.every(name => inputValues[name] !== '');

  if (featureNames.length === 0) {
    return <LoadingSpinner message="Loading feature information..." />;
  }

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Tumor Diagnosis
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Enter patient measurements for AI-assisted diagnosis
        </p>
      </div>

      {error && <ErrorAlert message={error} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="card bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Patient Measurements ({featureNames.length} features)
          </h3>
          <div className="max-h-96 overflow-y-auto">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {featureNames.map((name) => (
                <div key={name} className="flex flex-col gap-2">
                  <label
                    htmlFor={name}
                    className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                  >
                    {name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </label>
                  <input
                    type="number"
                    id={name}
                    name={name}
                    value={inputValues[name] || ''}
                    onChange={handleInputChange}
                    step="0.01"
                    placeholder="0.00"
                    className="block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  />
                </div>
              ))}
            </div>
          </div>
          <div className="flex gap-4 mt-6">
            <button
              onClick={handleLoadSampleData}
              className="flex-1 flex justify-center items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-gray-600 hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
            >
              Load Sample Data
            </button>
            <button
              onClick={handlePredict}
              disabled={!isFormValid || isLoading}
              className="flex-1 flex justify-center items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
            >
              {isLoading ? (
                <>
                  <Loader className="w-4 h-4 animate-spin mr-2" />
                  Analyzing...
                </>
              ) : (
                'Analyze Tumor'
              )}
            </button>
          </div>
        </div>

        {prediction && (
          <div
            className={`card rounded-lg shadow-xl p-6 border-2 ${
              prediction.diagnosis === 'Malignant'
                ? 'border-red-400 bg-red-50 dark:bg-red-900/20'
                : 'border-green-400 bg-green-50 dark:bg-green-900/20'
            }`}
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Diagnosis Result
            </h3>
            <div className="space-y-4">
              <div
                className={`text-center p-6 rounded-lg text-2xl font-bold ${
                  prediction.diagnosis === 'Malignant'
                    ? 'bg-red-200 text-red-800 dark:bg-red-800 dark:text-red-200'
                    : 'bg-green-200 text-green-800 dark:bg-green-800 dark:text-green-200'
                }`}
              >
                {prediction.diagnosis.toUpperCase()}
              </div>
              <div className="confidence-container">
                <div className="flex justify-between text-sm font-medium text-gray-700 dark:text-gray-300">
                  <span>Confidence Score</span>
                  <span>{prediction.confidence.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5 mt-2">
                  <div
                    className="h-2.5 rounded-full transition-all duration-500 bg-indigo-600"
                    style={{ width: `${prediction.confidence}%` }}
                  ></div>
                </div>
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                <p><strong>Model Accuracy:</strong> {prediction.accuracy.toFixed(1)}%</p>
                {prediction.prediction_timestamp && (
                  <p><strong>Prediction Time:</strong> {new Date(prediction.prediction_timestamp).toLocaleString()}</p>
                )}
              </div>
              <div className="disclaimer bg-gray-100 dark:bg-gray-700 p-4 rounded-md border-l-4 border-yellow-500 text-sm text-gray-600 dark:text-gray-400">
                <strong>Medical Disclaimer:</strong> This AI system is for educational purposes only.
                Always consult with qualified medical professionals for actual diagnosis and treatment.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// --- Data Explorer Tab Component
const DataExplorerTab = () => {
  const [data, setData] = useState<DataRow[]>([]);
  const [featureNames, setFeatureNames] = useState<string[]>([]);
  const [filter, setFilter] = useState('');
  const [limit, setLimit] = useState(100);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalCount, setTotalCount] = useState(0);

  const fetchData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams();
      if (filter) params.append('diagnosis', filter);
      if (limit > 0) params.append('limit', limit.toString());
      
      const url = `http://127.0.0.1:5000/data?${params}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch data: ${response.status}`);
      }
      
      const result: DataResponse = await response.json();
      setData(result.data || []);
      setTotalCount(result.total_count || 0);
      setFeatureNames(result.feature_names || []);
    } catch (error) {
      console.error("Failed to fetch data:", error);
      setError(error instanceof Error ? error.message : 'Failed to fetch data');
      setData([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [filter, limit]);

  const handleFilterChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setFilter(e.target.value);
  };

  const handleLimitChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setLimit(parseInt(e.target.value));
  };

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Dataset Explorer
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Explore the Breast Cancer Dataset ({totalCount} total samples)
        </p>
      </div>

      {error && <ErrorAlert message={error} onRetry={fetchData} />}

      <div className="card bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-4 gap-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Sample Data ({data.length} showing)
          </h3>
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600 dark:text-gray-400">
                Filter by Diagnosis:
              </label>
              <select
                value={filter}
                onChange={handleFilterChange}
                className="rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              >
                <option value="">All</option>
                <option value="M">Malignant</option>
                <option value="B">Benign</option>
              </select>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600 dark:text-gray-400">
                Show:
              </label>
              <select
                value={limit}
                onChange={handleLimitChange}
                className="rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              >
                <option value={50}>50 rows</option>
                <option value={100}>100 rows</option>
                <option value={200}>200 rows</option>
                <option value={0}>All rows</option>
              </select>
            </div>
          </div>
        </div>
        
        <div className="data-table-container overflow-x-auto max-h-96">
          {isLoading ? (
            <LoadingSpinner message="Loading data..." />
          ) : (
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-700 sticky top-0">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Diagnosis
                  </th>
                  {featureNames.slice(0, 4).map((name) => (
                    <th key={name} className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      {name.replace(/_/g, ' ')}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {data.map((row, index) => (
                  <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <span
                        className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                          row.diagnosis === 'M'
                            ? 'bg-red-100 text-red-800 dark:bg-red-800 dark:text-red-200'
                            : 'bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-200'
                        }`}
                      >
                        {row.diagnosis === 'M' ? 'Malignant' : 'Benign'}
                      </span>
                    </td>
                    {featureNames.slice(0, 4).map((name) => (
                      <td key={name} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                        {row[name] !== null && row[name] !== undefined
                          ? typeof row[name] === 'number'
                            ? (row[name] as number).toFixed(3)
                            : row[name]
                          : 'N/A'
                        }
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        
        {data.length === 0 && !isLoading && (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            No data available
          </div>
        )}
      </div>
    </div>
  );
};

// --- Main App Component
function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: <Grid /> },
    { id: 'train', label: 'Model Training', icon: <Brain /> },
    { id: 'predict', label: 'Diagnosis', icon: <Activity /> },
    { id: 'data', label: 'Data Explorer', icon: <Database /> },
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <DashboardTab />;
      case 'train':
        return <TrainModelTab />;
      case 'predict':
        return <DiagnosisTab />;
      case 'data':
        return <DataExplorerTab />;
      default:
        return <DashboardTab />;
    }
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors duration-300">
        <header className="bg-white dark:bg-gray-800 shadow-sm sticky top-0 z-10 transition-colors duration-300">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col sm:flex-row justify-between items-center">
            <div className="flex items-center gap-3">
              <BarChart2 className="w-8 h-8 text-indigo-600 dark:text-indigo-400" />
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                Medical AI
              </h1>
            </div>
            <nav className="flex flex-wrap sm:flex-nowrap gap-2 mt-4 sm:mt-0">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ease-in-out
                    ${
                      activeTab === item.id
                        ? 'bg-indigo-600 text-white shadow-md'
                        : 'bg-transparent text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-900 dark:hover:text-white'
                    }`}
                >
                  {item.icon}
                  {item.label}
                </button>
              ))}
            </nav>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {renderTabContent()}
        </main>
      </div>
    </ErrorBoundary>
  );
}

export default App;