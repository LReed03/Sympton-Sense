using Python.Runtime;

namespace SymptomSense.Backend.Services
{
    public class MachineLearningService
    {
        public string PredictDisease(List<string> symptoms)
        {
            using (Py.GIL())
            {
                string pythonPath = Path.GetFullPath(
                    Path.Combine(
                        Directory.GetCurrentDirectory(),
                        "..",
                        "ml",
                        "src"
                    )
                );

                dynamic sys = Py.Import("sys");
                sys.path.append(pythonPath);

                dynamic predictModule = Py.Import("predict");

                using PyObject pythonSymptoms = symptoms.ToPython();
                using PyObject result =
                    predictModule.predict_disease(pythonSymptoms);

                return result.ToString();
            }
        }
    }
}