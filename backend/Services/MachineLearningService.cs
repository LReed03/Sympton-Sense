using Python.Runtime;
namespace SymptomSense.Backend.Services
{
    public class MachineLearningService
    {     
        public string PredictDisease(List<string> symptoms)
        {
            PythonEngine.Initialize();

            try
            {
                dynamic sys = Py.Import("sys");
                sys.path.append("../ml/src");

            }
            finally
            {
                PythonEngine.Shutdown();
            }
            return "Common Cold";
        }
    }
}