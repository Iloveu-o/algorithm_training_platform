import axios from 'axios';

const api = axios.create({
  baseURL: '/api', // Vite proxy will forward this to http://127.0.0.1:8000
  timeout: 10000,
});

export const getBatteryAnalysis = async (batteryId) => {
  const response = await api.get(`/battery/${batteryId}`, {
    params: { plots: true }
  });
  return response.data;
};

export const startUnifiedTraining = async (payload) => {
  const response = await api.post('/train/unified', payload);
  return response.data;
};

export const getUnifiedTrainingJob = async (jobId) => {
  const response = await api.get(`/train/unified/${jobId}`);
  return response.data;
};

export const cancelUnifiedTrainingJob = async (jobId) => {
  const response = await api.post(`/train/unified/${jobId}/cancel`);
  return response.data;
};

export const listPredictModels = async () => {
  const response = await api.get('/predict/models');
  return response.data;
};

export const runPredict = async (payload, options = {}) => {
  const response = await api.post('/predict/run', payload, {
    signal: options.signal,
  });
  return response.data;
};
