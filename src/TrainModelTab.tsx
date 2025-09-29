import { useState, useEffect, useRef } from 'react';
import { Brain, UploadCloud, Trash2, FileQuestion, Loader, CheckCircle, AlertCircle, Database, Target, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Info } from 'lucide-react';
import {
  uploadTrainingFile,
  getUploadedFiles,
  deleteUploadedFile,
  analyzeUploadedFile,
  validateCsvFile,
  retrainModel,
  getDataset
} from './apiService';

interface UploadedFile {
  filename: string;
  upload_time: string;
  size: number;
  columns?: string[];
  total_rows?: number;
}

interface FileAnalysis {
  filename: string;
  columns: string[];
  total_rows: number;
  sample_data: any[];
  column_types: Record<string, string>;
  missing_values: Record<string, number>;
}

interface DataRow {
  [key: string]: number | string | null;
}

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

export const TrainModelTab = () => {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [dataPreview, setDataPreview] = useState<DataRow[]>([]);
  const [featureNames, setFeatureNames] = useState<string[]>([]);
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);
  const [analysis, setAnalysis] = useState<FileAnalysis | null>(null);
  
  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [rowsPerPage, setRowsPerPage] = useState(50);
  
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [training, setTraining] = useState(false);
  
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [trainingResult, setTrainingResult] = useState<any>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchFiles();
    fetchDataPreview();
  }, []);

  const fetchFiles = async () => {
    setLoadingFiles(true);
    setUploadError(null);
    try {
      const result = await getUploadedFiles();
      setUploadedFiles(result.files);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Failed to fetch files');
    } finally {
      setLoadingFiles(false);
    }
  };

  const fetchDataPreview = async () => {
    setLoadingPreview(true);
    setPreviewError(null);
    try {
      const result = await getDataset(undefined, 0);
      setDataPreview(result.data || []);
      setFeatureNames(result.feature_names || []);
      setCurrentPage(1);
    } catch (error) {
      setPreviewError(error instanceof Error ? error.message : 'Failed to fetch data preview');
    } finally {
      setLoadingPreview(false);
    }
  };

  const handleFileChange = async (file: File | null) => {
    if (!file) return;

    setUploading(true);
    setUploadError(null);
    setUploadMessage(null);

    const validation = await validateCsvFile(file);
    if (!validation.valid) {
      setUploadError(validation.errors.join('. '));
      setUploading(false);
      return;
    }

    try {
      const result = await uploadTrainingFile(file);
      if (result.success) {
        setUploadMessage(`✓ ${result.message}`);
        fetchFiles();
        fetchDataPreview();
      } else {
        setUploadError(result.message);
      }
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Upload failed');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDeleteFile = async (filename: string) => {
    if (!window.confirm(`Delete "${filename}"?`)) return;
    
    try {
      await deleteUploadedFile(filename);
      fetchFiles();
      if (analysis?.filename === filename) setAnalysis(null);
      setUploadMessage(`File "${filename}" deleted.`);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Failed to delete file');
    }
  };

  const handleAnalyzeFile = async (filename: string) => {
    setLoadingAnalysis(true);
    setAnalysis(null);
    setAnalysisError(null);
    try {
      const result = await analyzeUploadedFile(filename);
      setAnalysis(result);
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : 'Analysis failed');
    } finally {
      setLoadingAnalysis(false);
    }
  };

  const handleRetrain = async () => {
    setTraining(true);
    setTrainingError(null);
    setTrainingResult(null);
    
    try {
      const result = await retrainModel(selectedIndices.length > 0 ? selectedIndices : undefined);
      setTrainingResult(result);
      fetchDataPreview();
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : 'Training failed');
    } finally {
      setTraining(false);
    }
  };

  const handleSelectAll = () => {
    const allIndices = dataPreview.map((_, idx) => idx);
    setSelectedIndices(allIndices);
  };

  const handleSelectPageIndices = () => {
    const pageIndices = paginatedData.map((_, idx) => startIndex + idx);
    setSelectedIndices(prev => {
      const newSet = new Set(prev);
      pageIndices.forEach(idx => newSet.add(idx));
      return Array.from(newSet);
    });
  };

  const handleClearSelection = () => setSelectedIndices([]);

  const handleToggleRow = (index: number) => {
    setSelectedIndices(prev => 
      prev.includes(index) 
        ? prev.filter(x => x !== index) 
        : [...prev, index]
    );
  };

  // Pagination calculations
  const totalPages = Math.ceil(dataPreview.length / rowsPerPage);
  const startIndex = (currentPage - 1) * rowsPerPage;
  const endIndex = Math.min(startIndex + rowsPerPage, dataPreview.length);
  const paginatedData = dataPreview.slice(startIndex, endIndex);

  // Selection summary
  const selectedMalignant = selectedIndices.filter(idx => dataPreview[idx]?.diagnosis === 'M').length;
  const selectedBenign = selectedIndices.filter(idx => dataPreview[idx]?.diagnosis === 'B').length;

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Model Training & Data Management</h1>
        <p className="text-lg text-gray-600">Upload datasets, preview data, and retrain the prediction model</p>
      </div>

      {uploadMessage && <SuccessAlert message={uploadMessage} />}
      {uploadError && <ErrorAlert message={uploadError} />}
      {trainingError && <ErrorAlert message={trainingError} />}

      {trainingResult && (
        <div className="medical-card p-6 bg-gradient-to-r from-medical-success-50 to-medical-info-50 border-2 border-medical-success-300">
          <h3 className="section-header text-xl">
            <CheckCircle className="w-6 h-6 text-medical-success-600" />
            Training Complete
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="metric-display">
              <span className="text-sm text-gray-600">New Accuracy</span>
              <span className="text-2xl font-bold text-medical-success-600">{trainingResult.new_accuracy.toFixed(1)}%</span>
            </div>
            <div className="metric-display">
              <span className="text-sm text-gray-600">Samples Used</span>
              <span className="text-2xl font-bold text-medical-primary-600">{trainingResult.samples_used}</span>
            </div>
            <div className="metric-display">
              <span className="text-sm text-gray-600">Precision</span>
              <span className="text-2xl font-bold text-purple-600">{(trainingResult.metrics?.precision * 100).toFixed(1)}%</span>
            </div>
            <div className="metric-display">
              <span className="text-sm text-gray-600">Recall</span>
              <span className="text-2xl font-bold text-orange-600">{(trainingResult.metrics?.recall * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="medical-card p-6 space-y-4">
          <h3 className="section-header text-xl">
            <UploadCloud className="w-6 h-6 text-medical-primary-500" />
            Upload Dataset
          </h3>
          
          <div
            className="upload-zone"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handleFileChange(e.dataTransfer.files[0]);
            }}
          >
            <UploadCloud className="w-12 h-12 text-gray-400 mb-2" />
            <p className="text-gray-600 font-medium">Drop CSV file or click to browse</p>
            <p className="text-xs text-gray-500 mt-1">Max 10MB • Must include 'diagnosis' column</p>
            <input ref={fileInputRef} type="file" accept=".csv" onChange={(e) => handleFileChange(e.target.files?.[0] || null)} className="hidden" disabled={uploading} />
          </div>

          {uploading && <LoadingSpinner message="Processing file..." />}

          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-2">Uploaded Files ({uploadedFiles.length})</h4>
            {loadingFiles ? <LoadingSpinner message="Loading files..." /> : (
              <div className="space-y-2 max-h-64 overflow-y-auto scrollbar-medical">
                {uploadedFiles.map((file) => (
                  <div key={file.filename} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{file.filename}</p>
                      <p className="text-xs text-gray-500">{file.total_rows || 0} rows • {(file.size / 1024).toFixed(1)} KB</p>
                    </div>
                    <div className="flex gap-2 ml-4">
                      <button onClick={() => handleAnalyzeFile(file.filename)} className="text-medical-info-600 hover:text-medical-info-800 transition-colors" title="Analyze">
                        <FileQuestion className="w-5 h-5" />
                      </button>
                      <button onClick={() => handleDeleteFile(file.filename)} className="text-medical-danger-600 hover:text-medical-danger-800 transition-colors" title="Delete">
                        <Trash2 className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="medical-card p-6 space-y-4">
          <h3 className="section-header text-xl">
            <Database className="w-6 h-6 text-medical-primary-500" />
            Dataset Analysis
          </h3>
          
          {loadingAnalysis ? <LoadingSpinner message="Analyzing..." /> : analysis ? (
            <div className="space-y-4">
              <div className="bg-medical-info-50 rounded-lg p-4">
                <p className="text-sm font-medium text-gray-700 mb-2">{analysis.filename}</p>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div><span className="text-gray-600">Rows:</span><span className="font-bold ml-2">{analysis.total_rows}</span></div>
                  <div><span className="text-gray-600">Columns:</span><span className="font-bold ml-2">{analysis.columns.length}</span></div>
                </div>
              </div>
              
              <div className="max-h-64 overflow-y-auto scrollbar-medical">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 sticky top-0">
                    <tr>
                      <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700">Column</th>
                      <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700">Type</th>
                      <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700">Missing</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {analysis.columns.map((col) => (
                      <tr key={col} className="hover:bg-gray-50">
                        <td className="px-3 py-2 font-medium text-gray-900">{col}</td>
                        <td className="px-3 py-2 text-gray-600">{analysis.column_types[col]}</td>
                        <td className="px-3 py-2">
                          <span className={analysis.missing_values[col] > 0 ? 'text-medical-danger-600 font-semibold' : 'text-gray-600'}>
                            {analysis.missing_values[col]}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : analysisError ? <ErrorAlert message={analysisError} /> : (
            <div className="text-center text-gray-500 py-12">
              <FileQuestion className="w-16 h-16 text-gray-300 mx-auto mb-3" />
              <p>Select a file to view analysis</p>
            </div>
          )}
        </div>
      </div>

      <div className="medical-card p-6">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 gap-4">
          <div>
            <h3 className="section-header text-xl mb-2">
              <Target className="w-6 h-6 text-medical-primary-500" />
              Training Data Preview
            </h3>
            {selectedIndices.length > 0 && (
              <div className="flex gap-4 text-sm">
                <span className="text-gray-600">Selected: <span className="font-bold text-medical-primary-600">{selectedIndices.length}</span> total</span>
                <span className="text-gray-600">Malignant: <span className="font-bold text-medical-danger-600">{selectedMalignant}</span></span>
                <span className="text-gray-600">Benign: <span className="font-bold text-medical-success-600">{selectedBenign}</span></span>
              </div>
            )}
          </div>
          
          <div className="flex flex-wrap gap-2">
            <button onClick={handleSelectPageIndices} className="btn-medical-secondary text-sm py-2 px-4">Select Page</button>
            <button onClick={handleSelectAll} className="btn-medical-secondary text-sm py-2 px-4">Select All</button>
            <button onClick={handleClearSelection} className="btn-medical-secondary text-sm py-2 px-4">Clear</button>
            <button onClick={handleRetrain} disabled={training} className="btn-medical-primary text-sm py-2 px-4">
              {training ? <><Loader className="w-4 h-4 animate-spin" />Training...</> : <><Brain className="w-4 h-4" />Retrain{selectedIndices.length > 0 ? ` (${selectedIndices.length})` : ''}</>}
            </button>
          </div>
        </div>

        {previewError && <ErrorAlert message={previewError} onRetry={fetchDataPreview} />}

        {loadingPreview ? <LoadingSpinner message="Loading data..." /> : (
          <>
            <div className="data-table-container max-h-96">
              <table>
                <thead>
                  <tr>
                    <th className="w-12">
                      <input 
                        type="checkbox" 
                        className="checkbox-medical" 
                        checked={paginatedData.every((_, idx) => selectedIndices.includes(startIndex + idx))} 
                        onChange={(e) => {
                          if (e.target.checked) {
                            handleSelectPageIndices();
                          } else {
                            const pageIndices = paginatedData.map((_, idx) => startIndex + idx);
                            setSelectedIndices(prev => prev.filter(idx => !pageIndices.includes(idx)));
                          }
                        }}
                      />
                    </th>
                    <th>Index</th>
                    <th>ID</th>
                    <th>Diagnosis</th>
                    {featureNames.slice(0, 4).map((name) => <th key={name}>{name.replace(/_/g, ' ')}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {paginatedData.map((row, pageIdx) => {
                    const actualIdx = startIndex + pageIdx;
                    const isSelected = selectedIndices.includes(actualIdx);
                    return (
                      <tr 
                        key={actualIdx}
                        className={isSelected ? 'bg-medical-primary-50 border-l-4 border-medical-primary-500' : ''}
                      >
                        <td>
                          <input 
                            type="checkbox" 
                            className="checkbox-medical" 
                            checked={isSelected} 
                            onChange={() => handleToggleRow(actualIdx)} 
                          />
                        </td>
                        <td className="font-medium text-gray-600">{actualIdx}</td>
                        <td className="font-mono text-xs text-gray-500">{row.id ?? 'N/A'}</td>
                        <td><span className={row.diagnosis === 'M' ? 'badge-malignant' : 'badge-benign'}>{row.diagnosis === 'M' ? 'Malignant' : 'Benign'}</span></td>
                        {featureNames.slice(0, 4).map((name) => <td key={name}>{typeof row[name] === 'number' ? row[name].toFixed(3) : row[name] ?? 'N/A'}</td>)}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            <div className="flex flex-col sm:flex-row items-center justify-between mt-4 gap-4">
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600">Rows per page:</label>
                <select 
                  value={rowsPerPage} 
                  onChange={(e) => {
                    setRowsPerPage(Number(e.target.value));
                    setCurrentPage(1);
                  }} 
                  className="select-medical py-1.5 text-sm"
                >
                  <option value={25}>25</option>
                  <option value={50}>50</option>
                  <option value={100}>100</option>
                  <option value={200}>200</option>
                </select>
                <span className="text-sm text-gray-600 ml-4">
                  Showing {startIndex + 1}-{endIndex} of {dataPreview.length}
                </span>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setCurrentPage(1)}
                  disabled={currentPage === 1}
                  className="p-2 rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title="First page"
                >
                  <ChevronsLeft className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                  disabled={currentPage === 1}
                  className="p-2 rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title="Previous page"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                
                <span className="px-4 py-2 text-sm font-medium text-gray-700">
                  Page {currentPage} of {totalPages}
                </span>
                
                <button
                  onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                  disabled={currentPage === totalPages}
                  className="p-2 rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title="Next page"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setCurrentPage(totalPages)}
                  disabled={currentPage === totalPages}
                  className="p-2 rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title="Last page"
                >
                  <ChevronsRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};