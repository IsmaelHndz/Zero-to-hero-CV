import { useState } from "react";

function App() {
  const [description, setDescription] = useState("");
  const [cvFile, setCvFile] = useState<File | null>(null);
  const [response, setResponse] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!description || !cvFile) {
      alert("Please load all the information.");
      return;
    }
    setLoading(true);
    setResponse(null);

    // 1. Crea un archivo de texto en memoria con la descripción del empleo
    const jobDescriptionFile = new File(
      [description],
      "job_description.txt",
      { type: "text/plain" }
    );

    // 2. Prepara el form data igual que en el script de Python
    const formData = new FormData();
    formData.append("cv", cvFile);
    formData.append("job_position", jobDescriptionFile);

    try {
      const res = await fetch("http://127.0.0.1:5000/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        setResponse(`Error del servidor: ${res.status}\n${text}`);
      } else {
        const data = await res.json();
        setResponse(JSON.stringify(data, null, 2));
      }
    } catch (err) {
      setResponse("Error al enviar los datos: " + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <form
        className="bg-white p-8 rounded shadow-md w-full max-w-md space-y-4"
        onSubmit={handleSubmit}
      >
        <h1 className="text-2xl font-bold mb-4 text-center">Procesar CV</h1>

        <div>
          <label className="block mb-1 font-medium">Descripción del empleo</label>
          <textarea
            className="w-full border rounded px-3 py-2 focus:outline-none focus:ring"
            rows={4}
            value={description}
            onChange={e => setDescription(e.target.value)}
            required
          />
        </div>

        <div>
          <label className="block mb-1 font-medium">Cargar CV</label>
          <input
            type="file"
            accept=".pdf,.doc,.docx,.txt"
            onChange={e => setCvFile(e.target.files ? e.target.files[0] : null)}
            className="w-full"
            required
          />
        </div>

        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 transition"
          disabled={loading}
        >
          {loading ? "Enviando..." : "Enviar"}
        </button>

        {response && (
          <pre className="bg-gray-200 p-4 rounded mt-4 text-sm overflow-x-auto max-h-96">
            {response}
          </pre>
        )}
      </form>
    </div>
  );
}

export default App;