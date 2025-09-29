/* eslint-disable no-irregular-whitespace */
import { useState, useEffect, useRef } from 'react';
import { Brain, BarChart2, Activity, Database, Grid, AlertCircle, CheckCircle, Loader, TrendingUp, Users, Target, Award, Info, HelpCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell } from 'recharts';
import { predictWithApi, PredictionResult, getModelStatus, getDataset, getRandomSample, batchPredictFromFile } from './apiService';
import { TrainModelTab } from './TrainModelTab';

// Add CSS animation inline
const style = document.createElement('style');
style.textContent = `
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(-4px); }
    to { opacity: 1; transform: translateY(0); }
  }
`;
document.head.appendChild(style);

interface DataRow {
  [key: string]: number | string | null;
}

interface ModelStatus {
  accuracy: number;
  trained: boolean;
  total_samples: number;
  feature_count: number;
  feature_names?: string[];
  current_dataset?: { source: string; filename: string };
  metrics?: {
    test_accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    test_roc_auc: number;
  };
  feature_importance?: Array<{ feature: string; importance: number; rank: number }>;
  confusion_matrix?: {
    matrix: number[][];
    labels: string[];
    values: { true_negatives: number; false_positives: number; false_negatives: number; true_positives: number };
  };
}

// Feature descriptions for medical context
const FEATURE_DESCRIPTIONS: Record<string, string> = {
  'radius_mean': 'Average distance from center to perimeter points on the cell nucleus',
  'texture_mean': 'Standard deviation of gray-scale values, indicating surface texture variation',
  'perimeter_mean': 'Average perimeter measurement of the cell nucleus',
  'area_mean': 'Average area of the cell nucleus',
  'smoothness_mean': 'Local variation in radius lengths, measuring surface smoothness',
  'compactness_mean': 'Perimeter² / area - 1.0, indicating shape compactness',
  'concavity_mean': 'Severity of concave portions of the cell contour',
  'concave points_mean': 'Number of concave portions of the cell contour',
  'symmetry_mean': 'Symmetry of the cell nucleus',
  'fractal_dimension_mean': 'Coastline approximation - 1, measuring boundary complexity'
};

// Metric explanations for tooltips
const METRIC_INFO = {
  accuracy: {
    title: 'Model Accuracy',
    description: 'Percentage of correct predictions (both malignant and benign) out of all test cases. Higher is better.',
    interpretation: 'Shows overall model performance across all predictions.'
  },
  precision: {
    title: 'Precision (Positive Predictive Value)',
    description: 'Of all cases predicted as malignant, what percentage were actually malignant.',
    interpretation: 'High precision means fewer false alarms (benign cases wrongly flagged as malignant).'
  },
  recall: {
    title: 'Recall (Sensitivity)',
    description: 'Of all actual malignant cases, what percentage did the model correctly identify.',
    interpretation: 'High recall means fewer missed cancer cases (malignant cases wrongly labeled as benign).'
  },
  f1_score: {
    title: 'F1 Score',
    description: 'Harmonic mean of precision and recall, balancing both metrics.',
    interpretation: 'Useful when you need a balance between precision and recall. Ranges from 0-100%.'
  },
  roc_auc: {
    title: 'ROC-AUC Score',
    description: 'Area Under the Receiver Operating Characteristic curve. Measures model ability to distinguish between classes.',
    interpretation: 'Score of 100% = perfect classifier, 50% = random guessing. Higher is better.'
  }
};

// Tooltip component with smart positioning
const Tooltip = ({ content, children }: { content: { title: string; description: string; interpretation?: string }, children: React.ReactNode }) => {
  const [show, setShow] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const TOOLTIP_MARGIN = 8; // Minimum spacing from viewport edges

  const handleMouseEnter = () => {
    if (triggerRef.current) {
      const triggerRect = triggerRef.current.getBoundingClientRect();
      // Use fallback dimensions if the tooltip is not yet rendered
      const tooltipRect = tooltipRef.current
        ? tooltipRef.current.getBoundingClientRect()
        : { width: 320, height: 150 };

      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;

      // Default position: below the trigger
      let top = triggerRect.bottom + TOOLTIP_MARGIN;
      let left = triggerRect.left;

      // Adjust horizontally to prevent overflow
      if (left + tooltipRect.width > viewportWidth - TOOLTIP_MARGIN) {
        // If it overflows the right, move it left
        left = viewportWidth - tooltipRect.width - TOOLTIP_MARGIN;
      }
      if (left < TOOLTIP_MARGIN) {
        // If it overflows the left, clamp it to the left edge
        left = TOOLTIP_MARGIN;
      }

      // Adjust vertically to prevent overflow
      if (top + tooltipRect.height > viewportHeight - TOOLTIP_MARGIN) {
        // If it overflows the bottom, position it above the trigger
        top = triggerRect.top - tooltipRect.height - TOOLTIP_MARGIN;
      }

      // Final clamp to ensure it doesn't go off the top edge
      if (top < TOOLTIP_MARGIN) {
        top = TOOLTIP_MARGIN;
      }

      // Set position relative to the viewport (fixed positioning)
      setPosition({ top, left });
    }
    setShow(true);
  };

  return (
    <div
      className="relative inline-block"
      ref={triggerRef}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <div
          ref={tooltipRef}
          className="fixed z-[9999] w-80 max-w-[calc(100vw-32px)] bg-gray-900 text-white text-sm rounded-lg shadow-2xl p-4 pointer-events-none"
          style={{
            top: `${position.top}px`,
            left: `${position.left}px`,
            animation: 'fadeIn 0.15s ease-out'
          }}
        >
          <div className="font-bold mb-2 text-medical-primary-300">{content.title}</div>
          <div className="mb-2 leading-relaxed">{content.description}</div>
          {content.interpretation && (
            <div className="text-xs text-gray-300 border-t border-gray-700 pt-2 mt-2">
              <span className="font-semibold">Clinical Interpretation:</span> {content.interpretation}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const ErrorBoundary = ({ children, fallback = <div className="alert-error"><AlertCircle className="w-5 h-5" /><span>Something went wrong. Please refresh.</span></div> }: { children: React.ReactNode; fallback?: React.ReactNode }) => {
  const [hasError, setHasError] = useState(false);
  useEffect(() => {
    const handleError = () => setHasError(true);
    window.addEventListener('error', handleError);
    return () => window.removeEventListener('error', handleError);
  }, []);
  return hasError ? <>{fallback}</> : <>{children}</>;
};

const LoadingSpinner = ({ message = 'Loading...' }: { message?: string }) => (
  <div className="flex flex-col items-center justify-center p-12">
    <Loader className="w-10 h-10 spinner-medical mb-3 text-medical-primary-500" />
    <p className="text-gray-600 font-medium">{message}</p>
  </div>
);

const ErrorAlert = ({ message, onRetry }: { message: string; onRetry?: () => void }) => (
  <div className="alert-error">
    <AlertCircle className="w-5 h-5 flex-shrink-0" />
    <div className="flex-1"><p className="text-medical-danger-700 font-medium">{message}</p></div>
    {onRetry && <button onClick={onRetry} className="text-medical-danger-600 hover:text-medical-danger-800 font-semibold underline ml-4">Retry</button>}
  </div>
);

const SuccessAlert = ({ message }: { message: string }) => (
  <div className="alert-success">
    <CheckCircle className="w-5 h-5 flex-shrink-0" />
    <p className="text-medical-success-700 font-medium">{message}</p>
  </div>
);

const DashboardTab = () => {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchModelStatus = async () => {
    try {
      setLoading(true);
      setError(null);
      const status = await getModelStatus();
      setModelStatus(status);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to fetch dashboard data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchModelStatus(); }, []);

  if (loading) return <LoadingSpinner message="Loading dashboard..." />;
  if (error) return <ErrorAlert message={error} onRetry={fetchModelStatus} />;

  const accuracy = modelStatus?.accuracy ? `${modelStatus.accuracy.toFixed(1)}%` : '--';
  const metrics = modelStatus?.metrics;
  const precision = metrics?.precision ? (metrics.precision * 100).toFixed(1) : '--';
  const recall = metrics?.recall ? (metrics.recall * 100).toFixed(1) : '--';
  const f1Score = metrics?.f1_score ? (metrics.f1_score * 100).toFixed(1) : '--';
  const rocAuc = metrics?.test_roc_auc ? (metrics.test_roc_auc * 100).toFixed(1) : '--';

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Breast Cancer Prediction Dashboard</h1>
        <p className="text-lg text-gray-600">AI-Powered Diagnostic Assistance System</p>
        {!modelStatus?.trained && <div className="mt-4"><ErrorAlert message="Model not trained. Upload data and train in Model Training tab." /></div>}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <p className="text-sm font-medium text-gray-600">Model Accuracy</p>
                {/* <Tooltip content={METRIC_INFO.accuracy}>
                  <HelpCircle className="w-4 h-4 text-gray-400 cursor-help hover:text-medical-primary-500 transition-colors" />
                </Tooltip> */}
              </div>
              <p className="text-3xl font-bold text-medical-primary-600">{accuracy}</p>
            </div>
            <div className="w-14 h-14 bg-gradient-medical rounded-xl flex items-center justify-center"><Target className="w-7 h-7 text-white" /></div>
          </div>
          <div className="mt-3 flex items-center text-sm text-medical-success-600"><TrendingUp className="w-4 h-4 mr-1" /><span className="font-medium">Optimal Performance</span></div>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <p className="text-sm font-medium text-gray-600">Precision</p>
                {/* <Tooltip content={METRIC_INFO.precision}>
                  <HelpCircle className="w-4 h-4 text-gray-400 cursor-help hover:text-medical-primary-500 transition-colors" />
                </Tooltip> */}
              </div>
              <p className="text-3xl font-bold text-medical-success-600">{precision}%</p>
            </div>
            <div className="w-14 h-14 bg-gradient-success rounded-xl flex items-center justify-center"><Award className="w-7 h-7 text-white" /></div>
          </div>
          <div className="mt-3 text-sm text-gray-500">Positive predictive value</div>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 mb-1">Training Samples</p>
              <p className="text-3xl font-bold text-medical-info-600">{modelStatus?.total_samples || 0}</p>
            </div>
            <div className="w-14 h-14 bg-medical-info-500 rounded-xl flex items-center justify-center"><Users className="w-7 h-7 text-white" /></div>
          </div>
          <div className="mt-3 text-sm text-gray-500">Dataset records</div>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 mb-1">Features Used</p>
              <p className="text-3xl font-bold text-purple-600">{modelStatus?.feature_count || 0}</p>
            </div>
            <div className="w-14 h-14 bg-purple-500 rounded-xl flex items-center justify-center"><BarChart2 className="w-7 h-7 text-white" /></div>
          </div>
          <div className="mt-3 text-sm text-gray-500">Top predictive features</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="medical-card p-5">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-600">Recall</span>
              <Tooltip content={METRIC_INFO.recall}>
                <HelpCircle className="w-4 h-4 text-gray-400 cursor-help hover:text-medical-primary-500 transition-colors" />
              </Tooltip>
            </div>
            <span className="text-2xl font-bold text-gray-900">{recall}%</span>
          </div>
          <div className="progress-bar-container"><div className="progress-bar bg-medical-success-500" style={{ width: `${recall}%` }}></div></div>
        </div>
        <div className="medical-card p-5">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-600">F1 Score</span>
              <Tooltip content={METRIC_INFO.f1_score}>
                <HelpCircle className="w-4 h-4 text-gray-400 cursor-help hover:text-medical-primary-500 transition-colors" />
              </Tooltip>
            </div>
            <span className="text-2xl font-bold text-gray-900">{f1Score}%</span>
          </div>
          <div className="progress-bar-container"><div className="progress-bar bg-medical-primary-500" style={{ width: `${f1Score}%` }}></div></div>
        </div>
        <div className="medical-card p-5">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-600">ROC-AUC</span>
              <Tooltip content={METRIC_INFO.roc_auc}>
                <HelpCircle className="w-4 h-4 text-gray-400 cursor-help hover:text-medical-primary-500 transition-colors" />
              </Tooltip>
            </div>
            <span className="text-2xl font-bold text-gray-900">{rocAuc}%</span>
          </div>
          <div className="progress-bar-container"><div className="progress-bar bg-purple-500" style={{ width: `${rocAuc}%` }}></div></div>
        </div>
      </div>

      {modelStatus?.feature_importance && modelStatus.feature_importance.length > 0 && (
        <div className="medical-card p-6">
          <h3 className="section-header"><TrendingUp className="w-5 h-5 text-medical-primary-500" />Feature Importance Analysis</h3>
          <p className="text-sm text-gray-600 mb-6">Top {modelStatus.feature_importance.length} predictive features ranked by statistical significance. Hover over bars for detailed information.</p>
          
          {/* Interactive Bar Chart */}
          <div className="mb-6 bg-gray-50 rounded-lg p-4">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={modelStatus.feature_importance}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis 
                  type="number" 
                  stroke="#6b7280"
                  tick={{ fontSize: 12 }}
                  label={{ value: 'Importance Score', position: 'insideBottom', offset: -5, style: { fontSize: 12, fill: '#6b7280' } }}
                />
                <YAxis 
                  type="category" 
                  dataKey="feature" 
                  stroke="#6b7280"
                  tick={{ fontSize: 11 }}
                  width={140}
                />
                <RechartsTooltip
                  cursor={{ fill: 'rgba(59, 130, 246, 0.1)' }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      const description = FEATURE_DESCRIPTIONS[data.feature] || 'Cell nucleus measurement';
                      return (
                        <div className="bg-gray-900 text-white p-4 rounded-lg shadow-xl text-sm max-w-xs">
                          <div className="font-bold text-medical-primary-300 mb-2">#{data.rank} {data.feature}</div>
                          <div className="text-gray-300 mb-3 text-xs leading-relaxed">{description}</div>
                          <div className="flex justify-between items-center pt-2 border-t border-gray-700">
                            <span className="text-gray-400 text-xs">Importance Score:</span>
                            <span className="text-medical-success-400 font-bold">{data.importance.toFixed(2)}</span>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="importance" radius={[0, 8, 8, 0]}>
                  {modelStatus.feature_importance.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={`hsl(${220 - index * 10}, 70%, ${50 + index * 2}%)`}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Collapsible Detailed Stats */}
          <details className="mt-4">
            <summary className="cursor-pointer text-sm font-semibold text-medical-primary-600 hover:text-medical-primary-700 mb-4 flex items-center gap-2">
              <Info className="w-4 h-4" />
              Show Statistical Details
            </summary>
            <div className="bg-blue-50 rounded-lg p-4 mt-3">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm mb-4">
                <div>
                  <span className="text-gray-600">Top Feature:</span>
                  <span className="font-bold text-gray-900 ml-2">{modelStatus.feature_importance[0].feature}</span>
                </div>
                <div>
                  <span className="text-gray-600">Max Score:</span>
                  <span className="font-bold text-gray-900 ml-2">{modelStatus.feature_importance[0].importance.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-gray-600">Total Features:</span>
                  <span className="font-bold text-gray-900 ml-2">{modelStatus.feature_importance.length}</span>
                </div>
              </div>
              <div className="space-y-2">
                {modelStatus.feature_importance.map((item, idx) => {
                  const description = FEATURE_DESCRIPTIONS[item.feature] || 'Cell nucleus measurement';
                  return (
                    <div key={idx} className="bg-white rounded p-3 text-xs">
                      <div className="font-semibold text-gray-800 mb-1">#{item.rank} {item.feature}</div>
                      <div className="text-gray-600">{description}</div>
                      <div className="text-medical-primary-600 font-medium mt-1">Score: {item.importance.toFixed(2)}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </details>
        </div>
      )}

      {modelStatus?.confusion_matrix && (
        <div className="medical-card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="section-header"><Grid className="w-5 h-5 text-medical-primary-500" />Confusion Matrix</h3>
            <Tooltip content={{ 
              title: 'Confusion Matrix', 
              description: 'Shows how the model performs on each class. Rows represent actual diagnosis, columns represent predicted diagnosis.',
              interpretation: 'Diagonal values (TN, TP) are correct predictions. Off-diagonal (FP, FN) are errors.'
            }}>
              <HelpCircle className="w-5 h-5 text-gray-400 cursor-help hover:text-medical-primary-500 transition-colors" />
            </Tooltip>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="grid grid-cols-2 gap-4 max-w-md mx-auto">
              <div className="bg-medical-success-50 border-2 border-medical-success-200 rounded-lg p-6 text-center">
                <div className="text-3xl font-bold text-medical-success-700">{modelStatus.confusion_matrix.values.true_negatives}</div>
                <div className="text-sm font-medium text-gray-600 mt-2">True Negatives</div>
                <div className="text-xs text-gray-500 mt-1">Correct Benign</div>
              </div>
              <div className="bg-medical-danger-50 border-2 border-medical-danger-200 rounded-lg p-6 text-center">
                <div className="text-3xl font-bold text-medical-danger-700">{modelStatus.confusion_matrix.values.false_positives}</div>
                <div className="text-sm font-medium text-gray-600 mt-2">False Positives</div>
                <div className="text-xs text-gray-500 mt-1">Benign as Malignant</div>
              </div>
              <div className="bg-medical-warning-50 border-2 border-medical-warning-300 rounded-lg p-6 text-center">
                <div className="text-3xl font-bold text-medical-warning-700">{modelStatus.confusion_matrix.values.false_negatives}</div>
                <div className="text-sm font-medium text-gray-600 mt-2">False Negatives</div>
                <div className="text-xs text-gray-500 mt-1">Malignant as Benign</div>
              </div>
              <div className="bg-medical-success-100 border-2 border-medical-success-300 rounded-lg p-6 text-center">
                <div className="text-3xl font-bold text-medical-success-800">{modelStatus.confusion_matrix.values.true_positives}</div>
                <div className="text-sm font-medium text-gray-600 mt-2">True Positives</div>
                <div className="text-xs text-gray-500 mt-1">Correct Malignant</div>
              </div>
            </div>
            <div className="flex flex-col justify-center space-y-4">
              <div className="bg-blue-50 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-gray-700 mb-2">Performance Summary</h4>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li className="flex justify-between"><span>Total Predictions:</span><span className="font-semibold">{modelStatus.confusion_matrix.values.true_negatives + modelStatus.confusion_matrix.values.false_positives + modelStatus.confusion_matrix.values.false_negatives + modelStatus.confusion_matrix.values.true_positives}</span></li>
                  <li className="flex justify-between"><span>Correct:</span><span className="font-semibold text-medical-success-600">{modelStatus.confusion_matrix.values.true_negatives + modelStatus.confusion_matrix.values.true_positives}</span></li>
                  <li className="flex justify-between"><span>Incorrect:</span><span className="font-semibold text-medical-danger-600">{modelStatus.confusion_matrix.values.false_positives + modelStatus.confusion_matrix.values.false_negatives}</span></li>
                </ul>
              </div>
              <div className="alert-info"><AlertCircle className="w-5 h-5 flex-shrink-0" /><p className="text-sm text-medical-info-700">Matrix shows model performance in distinguishing benign vs malignant cases.</p></div>
              <div className="bg-medical-warning-50 border border-medical-warning-300 rounded-lg p-3">
                <p className="text-xs text-gray-700"><span className="font-semibold">Clinical Note:</span> False negatives (missed cancers) are typically more concerning than false positives in diagnostic contexts.</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {modelStatus?.feature_names && (
        <div className="medical-card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="section-header"><CheckCircle className="w-5 h-5 text-medical-success-500" />Selected Features ({modelStatus.feature_names.length})</h3>
            <div className="alert-info inline-flex items-center gap-2 py-2 px-3">
              <Info className="w-4 h-4" />
              <span className="text-xs">Hover over features for detailed descriptions</span>
            </div>
          </div>
          <p className="text-sm text-gray-600 mb-4">Features used for breast cancer prediction. Provide accurate measurements when making predictions.</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {modelStatus.feature_names.map((feature, idx) => {
              const description = FEATURE_DESCRIPTIONS[feature] || 'Cell nucleus measurement used in diagnosis';
              return (
                <Tooltip key={idx} content={{ title: feature, description }}>
                  <div className="feature-tag cursor-help">
                    <div className="flex items-center gap-2">
                      <span className="w-6 h-6 bg-medical-primary-500 text-white rounded-full flex items-center justify-center text-xs font-bold">{idx + 1}</span>
                      <span className="font-medium">{feature}</span>
                      <Info className="w-4 h-4 text-gray-400 ml-auto opacity-70" />
                    </div>
                  </div>
                </Tooltip>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

// Diagnosis Tab Component
const DiagnosisTab = () => {
  const [featureNames, setFeatureNames] = useState<string[]>([]);
  const [inputValues, setInputValues] = useState<Record<string, string>>({});
  const [featureRanges, setFeatureRanges] = useState<Record<string, { min: number, max: number }>>({});
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingFeatures, setLoadingFeatures] = useState(true);
  const [batchPredicting, setBatchPredicting] = useState(false);
  const [batchResults, setBatchResults] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const fetchFeatureData = async () => {
      try {
        setLoadingFeatures(true);
        const status = await getModelStatus();
        const dataset = await getDataset();
        if (status.feature_names && dataset.data) {
          const names = status.feature_names;
          setFeatureNames(names);

          const initialValues: Record<string, string> = {};
          const ranges: Record<string, { min: number, max: number }> = {};
          names.forEach((name: string) => {
            initialValues[name] = '';
            const values = dataset.data.map(row => parseFloat(String(row[name]))).filter(v => !isNaN(v));
            if (values.length > 0) {
              ranges[name] = {
                min: Math.min(...values),
                max: Math.max(...values)
              };
            } else {
              ranges[name] = { min: 0, max: 0 };
            }
          });
          setInputValues(initialValues);
          setFeatureRanges(ranges);
        }
      } catch (error) {
        setError(error instanceof Error ? error.message : 'Failed to load features or dataset.');
      } finally {
        setLoadingFeatures(false);
      }
    };
    fetchFeatureData();
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setInputValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleLoadRandom = async () => {
    try {
      setIsLoading(true);
      const sample = await getRandomSample();
      const newValues: Record<string, string> = {};
      sample.feature_names.forEach((name, idx) => {
        newValues[name] = sample.features[idx].toString();
      });
      setInputValues(newValues);
      setPrediction(null);
      setError(null);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to load random sample');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePredict = async () => {
    setIsLoading(true);
    setPrediction(null);
    setError(null);
    try {
      const features = featureNames.map((name) => {
        const value = parseFloat(inputValues[name] || '0');
        return isNaN(value) ? 0 : value;
      });
      const result = await predictWithApi(features);
      setPrediction(result);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Prediction failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleBatchPredict = async (file: File | null) => {
    if (!file) return;
    setBatchPredicting(true);
    setBatchResults(null);
    setError(null);
    try {
      const result = await batchPredictFromFile(file);
      setBatchResults(result);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Batch prediction failed');
    } finally {
      setBatchPredicting(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const isFormValid = featureNames.every((name) => inputValues[name] !== '');

  if (loadingFeatures) return <LoadingSpinner message="Loading features..." />;

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Breast Cancer Risk Assessment</h1>
        <p className="text-lg text-gray-600">Enter patient measurements for AI-assisted prediction</p>
      </div>

      {error && <ErrorAlert message={error} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="medical-card p-6">
          <h3 className="section-header text-xl">Patient Measurements ({featureNames.length} features)</h3>
          <div className="max-h-96 overflow-y-auto scrollbar-medical pr-2">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {featureNames.map((name) => (
                <div key={name} className="space-y-2">
                  <label htmlFor={name} className="block text-sm font-semibold text-gray-700">{name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}</label>
                  <input
                    type="number"
                    id={name}
                    name={name}
                    value={inputValues[name] || ''}
                    onChange={handleInputChange}
                    step="0.01"
                    placeholder={featureRanges[name] ? `e.g., between ${featureRanges[name].min.toFixed(2)} and ${featureRanges[name].max.toFixed(2)}` : '0.00'}
                    className="input-medical"
                  />
                </div>
              ))}
            </div>
          </div>
          <div className="flex gap-4 mt-6">
            <button onClick={handleLoadRandom} disabled={isLoading} className="btn-medical-secondary flex-1">{isLoading ? <><Loader className="w-4 h-4 animate-spin" />Loading...</> : 'Load Random Sample'}</button>
            <button onClick={handlePredict} disabled={!isFormValid || isLoading} className="btn-medical-primary flex-1">{isLoading ? <><Loader className="w-4 h-4 animate-spin" />Analyzing...</> : <><Activity className="w-4 h-4" />Predict Risk</>}</button>
          </div>
        </div>

        {prediction ? (
          <div className={`medical-card p-6 border-2 ${prediction.diagnosis === 'Malignant' ? 'border-medical-danger-400 bg-medical-danger-50' : 'border-medical-success-400 bg-medical-success-50'}`}>
            <h3 className="section-header text-xl"><CheckCircle className="w-5 h-5" />Prediction Result</h3>
            <div className="space-y-4">
              <div className={`text-center p-8 rounded-xl text-3xl font-bold shadow-lg ${prediction.diagnosis === 'Malignant' ? 'bg-gradient-danger text-white' : 'bg-gradient-success text-white'}`}>{prediction.diagnosis.toUpperCase()}</div>
              <div className="bg-white rounded-lg p-4">
                <div className="flex justify-between text-sm font-semibold text-gray-700 mb-2">
                  <span>Confidence</span>
                  <span className="text-medical-primary-600">{prediction.confidence.toFixed(1)}%</span>
                </div>
                <div className="progress-bar-container"><div className="progress-bar bg-gradient-medical" style={{ width: `${prediction.confidence}%` }}></div></div>
              </div>
              <div className="bg-white rounded-lg p-4 space-y-2 text-sm">
                <div className="flex justify-between"><span className="text-gray-600">Model Accuracy:</span><span className="font-bold text-gray-900">{prediction.accuracy.toFixed(1)}%</span></div>
                {prediction.prediction_timestamp && <div className="flex justify-between"><span className="text-gray-600">Timestamp:</span><span className="font-medium text-gray-700">{new Date(prediction.prediction_timestamp).toLocaleString()}</span></div>}
              </div>
              <div className="alert-warning"><AlertCircle className="w-5 h-5 flex-shrink-0" /><div className="text-sm"><strong className="font-semibold">Disclaimer:</strong> For educational purposes only. Consult medical professionals for diagnosis.</div></div>
            </div>
          </div>
        ) : (
          <div className="medical-card p-6 flex flex-col items-center justify-center text-center">
            <Activity className="w-16 h-16 text-gray-300 mb-4" />
            <h3 className="text-lg font-semibold text-gray-700 mb-2">Ready for Analysis</h3>
            <p className="text-sm text-gray-500">Fill measurements and click "Predict Risk"</p>
          </div>
        )}
      </div>

      <div className="medical-card p-6">
        <h3 className="section-header text-xl"><Database className="w-5 h-5 text-medical-primary-500" />Batch Prediction from CSV</h3>
        <p className="text-sm text-gray-600 mb-4">Upload a CSV file with patient data (no diagnosis column needed). Missing values will be handled automatically.</p>
        <div className="flex items-center gap-4">
          <input ref={fileInputRef} type="file" accept=".csv" onChange={(e) => handleBatchPredict(e.target.files?.[0] || null)} className="input-medical flex-1" disabled={batchPredicting} />
          {batchPredicting && <Loader className="w-6 h-6 spinner-medical text-medical-primary-500" />}
        </div>

        {batchResults && (
          <div className="mt-6">
            <div className="bg-medical-success-50 rounded-lg p-4 mb-4">
              <p className="text-sm font-semibold text-gray-700">Successfully predicted {batchResults.total_predictions} samples</p>
            </div>
            <div className="data-table-container max-h-96">
              <table>
                <thead>
                  <tr>
                    <th>Row</th>
                    <th>Diagnosis</th>
                    <th>Confidence</th>
                    <th>Benign %</th>
                    <th>Malignant %</th>
                  </tr>
                </thead>
                <tbody>
                  {batchResults.predictions.map((pred: any, idx: number) => (
                    <tr key={idx}>
                      <td className="font-medium">{pred.row_index + 1}</td>
                      <td><span className={pred.diagnosis === 'Malignant' ? 'badge-malignant' : 'badge-benign'}>{pred.diagnosis}</span></td>
                      <td className="font-semibold">{pred.confidence.toFixed(1)}%</td>
                      <td>{(pred.probability_benign * 100).toFixed(1)}%</td>
                      <td>{(pred.probability_malignant * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

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
      const result = await getDataset(filter || undefined, limit > 0 ? limit : undefined);
      setData(result.data || []);
      setTotalCount(result.total_count || 0);
      setFeatureNames(result.feature_names || []);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to fetch data');
      setData([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, [filter, limit]);

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900">Dataset Explorer</h1>
        <p className="mt-2 text-gray-600">Browse breast cancer dataset ({totalCount} samples)</p>
      </div>

      {error && <ErrorAlert message={error} onRetry={fetchData} />}

      <div className="medical-card p-6">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-4 gap-4">
          <h3 className="text-lg font-semibold text-gray-900">Sample Data ({data.length} showing)</h3>
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600">Filter:</label>
              <select value={filter} onChange={(e) => setFilter(e.target.value)} className="select-medical">
                <option value="">All</option>
                <option value="M">Malignant</option>
                <option value="B">Benign</option>
              </select>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600">Show:</label>
              <select value={limit} onChange={(e) => setLimit(parseInt(e.target.value))} className="select-medical">
                <option value={50}>50</option>
                <option value={100}>100</option>
                <option value={200}>200</option>
                <option value={0}>All</option>
              </select>
            </div>
          </div>
        </div>

        {isLoading ? <LoadingSpinner message="Loading data..." /> : (
          <div className="data-table-container max-h-96">
            <table>
              <thead>
                <tr>
                  <th>Diagnosis</th>
                  {featureNames.slice(0, 4).map((name) => <th key={name}>{name.replace(/_/g, ' ')}</th>)}
                </tr>
              </thead>
              <tbody>
                {data.map((row, idx) => (
                  <tr key={idx}>
                    <td><span className={row.diagnosis === 'M' ? 'badge-malignant' : 'badge-benign'}>{row.diagnosis === 'M' ? 'Malignant' : 'Benign'}</span></td>
                    {featureNames.slice(0, 4).map((name) => <td key={name}>{row[name] !== null && row[name] !== undefined ? (typeof row[name] === 'number' ? (row[name] as number).toFixed(3) : row[name]) : 'N/A'}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {data.length === 0 && !isLoading && <div className="text-center py-8 text-gray-500">No data available</div>}
      </div>
    </div>
  );
};

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: <Grid /> },
    { id: 'train', label: 'Model Training', icon: <Brain /> },
    { id: 'predict', label: 'Diagnosis', icon: <Activity /> },
    { id: 'data', label: 'Data Explorer', icon: <Database /> }
  ];

  const renderTab = () => {
    switch (activeTab) {
      case 'dashboard': return <DashboardTab />;
      case 'train': return <TrainModelTab />;
      case 'predict': return <DiagnosisTab />;
      case 'data': return <DataExplorerTab />;
      default: return <DashboardTab />;
    }
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white shadow-sm sticky top-0 z-10">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col sm:flex-row justify-between items-center">
            <div className="flex items-center gap-3">
              <BarChart2 className="w-8 h-8 text-medical-primary-600" />
              <h1 className="text-2xl font-bold text-gray-900">Beacan Predicts</h1>
            </div>
            <nav className="flex flex-wrap gap-2 mt-4 sm:mt-0">
              {navItems.map((item) => (
                <button key={item.id} onClick={() => setActiveTab(item.id)} className={`tab-button ${activeTab === item.id ? 'tab-button-active' : 'tab-button-inactive'}`}>
                  {item.icon}
                  {item.label}
                </button>
              ))}
            </nav>
          </div>
        </header>
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">{renderTab()}</main>
      </div>
    </ErrorBoundary>
  );
}

export default App;