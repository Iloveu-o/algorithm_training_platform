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
