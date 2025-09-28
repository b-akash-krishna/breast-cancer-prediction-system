import { useState, useEffect } from 'react';
import {
  Brain,
  BarChart2,
  Activity,
  Database,
  Grid,
} from 'lucide-react';
import { predictWithApi, PredictionResult } from './apiService';

// Define the expected structure of a data row from the backend
interface DataRow {
  [key: string]: number | string;
}

// Define the expected structure of the model status response
interface ModelStatus {
  accuracy: number;
  trained: boolean;
  total_samples: number;
}

// --- Dashboard Tab Component
const DashboardTab = () => {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModelStatus = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/status');
        if (!response.ok) {
          throw new Error('Failed to fetch model status');
        }
        const status: ModelStatus = await response.json();
        setModelStatus(status);
      } catch (error) {
        console.error("Failed to fetch dashboard data:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchModelStatus();
  }, []);

  if (loading) {
    return <div className="text-center p-8">Loading dashboard...</div>;
  }

  const accuracy = modelStatus?.accuracy ? (modelStatus.accuracy * 100).toFixed(1) + '%' : '--';
  const totalSamples = modelStatus?.total_samples || 0;

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Breast Cancer Classification Dashboard
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          AI-powered diagnostic assistance for medical professionals
        </p>
      </div>
      <div className="stats-grid grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 flex items-center gap-4">
          <div className="w-12 h-12 flex items-center justify-center rounded-full bg-indigo-500 text-white">
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
            <div className="text-2xl font-bold">30</div>
            <div className="text-sm text-gray-500">Features</div>
          </div>
        </div>
      </div>
    </div>
  );
};


// --- Train Model Tab Component
const TrainModelTab = () => {
    return (
        <div className="p-8 bg-white dark:bg-gray-800 rounded-lg shadow-xl">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            Model Training Status
          </h2>
          <p>
            The model is now trained on the server when the Flask backend starts.
            We will update this tab to allow for dynamic training soon.
          </p>
        </div>
      );
};

// --- Diagnosis Tab Component
const DiagnosisTab = () => {
  const [inputValues, setInputValues] = useState({
    radiusMean: '',
    textureMean: '',
    perimeterMean: '',
    areaMean: '',
    smoothnessMean: '',
    compactnessMean: '',
  });
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    setInputValues((prev) => ({ ...prev, [id]: value }));
  };

  const handleLoadSampleData = () => {
    const sampleData = {
      radiusMean: '11.76',
      textureMean: '21.6',
      perimeterMean: '74.72',
      areaMean: '427.9',
      smoothnessMean: '0.08637',
      compactnessMean: '0.04966',
    };
    setInputValues(sampleData);
    setPrediction(null);
  };

  const handlePredict = async () => {
    setIsLoading(true);
    setPrediction(null);
    try {
      const features = Object.values(inputValues).map(Number);
      
      // Pad the features array to match the 30 features expected by the backend model.
      while (features.length < 30) {
        features.push(0);
      }

      const result = await predictWithApi(features);
      setPrediction(result);
    } catch (error) {
      console.error(error);
      alert('Prediction failed. Please ensure the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const isFormValid = Object.values(inputValues).every((val) => val !== '');

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

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="card bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Patient Measurements
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {Object.keys(inputValues).map((key) => (
              <div key={key} className="flex flex-col gap-2">
                <label
                  htmlFor={key}
                  className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                >
                  {key.charAt(0).toUpperCase() + key.slice(1)}
                </label>
                <input
                  type="number"
                  id={key}
                  value={inputValues[key as keyof typeof inputValues]}
                  onChange={handleInputChange}
                  step="0.01"
                  placeholder="0.00"
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                />
              </div>
            ))}
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
              {isLoading ? 'Analyzing...' : 'Analyze Tumor'}
            </button>
          </div>
        </div>

        {prediction && (
          <div
            className={`card rounded-lg shadow-xl p-6 border-2 ${
              prediction.diagnosis === 'Malignant'
                ? 'border-red-400 bg-red-50'
                : 'border-green-400 bg-green-50'
            }`}
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Diagnosis Result
            </h3>
            <div className="space-y-4">
              <div
                className={`text-center p-6 rounded-lg text-2xl font-bold ${
                  prediction.diagnosis === 'Malignant'
                    ? 'bg-red-200 text-red-800'
                    : 'bg-green-200 text-green-800'
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
  const [filter, setFilter] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        const url = filter
          ? `http://127.0.0.1:5000/data?diagnosis=${filter}`
          : 'http://127.0.0.1:5000/data';
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error('Failed to fetch data');
        }
        const fetchedData: DataRow[] = await response.json();
        setData(fetchedData);
      } catch (error) {
        console.error("Failed to fetch data:", error);
        setData([]);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, [filter]);

  const handleFilterChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setFilter(e.target.value);
  };

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Dataset Explorer
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Explore the Wisconsin Breast Cancer Dataset
        </p>
      </div>

      <div className="card bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Sample Data
          </h3>
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
        </div>
        <div className="data-table-container overflow-x-auto max-h-96">
          {isLoading ? (
            <div className="text-center p-4">Loading data...</div>
          ) : (
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-700 sticky top-0">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Diagnosis
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Radius Mean
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Texture Mean
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Perimeter Mean
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {data.map((row, index) => (
                  <tr key={index}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <span
                        className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                          row.diagnosis === 'M'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-green-100 text-green-800'
                        }`}
                      >
                        {row.diagnosis === 'M' ? 'Malignant' : 'Benign'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                      {Number(row.radiusMean).toFixed(2)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                      {Number(row.textureMean).toFixed(2)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                      {Number(row.perimeterMean).toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
};

// --- Main App Component
function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: <Grid /> },
    { id: 'train', label: 'Model Status', icon: <Brain /> },
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
  );
}

export default App;
