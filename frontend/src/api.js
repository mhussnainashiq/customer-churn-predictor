import axios from "axios";
const api=axios.create({baseURL:import.meta.env.VITE_API_URL||"http://127.0.0.1:8000",timeout:30000});
export const getDashboard=async()=>(await api.get("/api/dashboard")).data;
export const getHistory=async()=>(await api.get("/api/customers?limit=200")).data;
export const predictCustomer=async(data)=>(await api.post("/api/predict",data)).data;
export const clearHistory=async()=>(await api.delete("/api/customers/history")).data;
export const uploadCsv=async(file)=>{const form=new FormData();form.append("file",file);return api.post("/api/predict/batch",form,{responseType:"blob"});};
export const getHealth=async()=>(await api.get("/api/health")).data;
