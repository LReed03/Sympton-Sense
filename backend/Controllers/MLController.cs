using Microsoft.AspNetCore.Mvc;
using SymptomSense.Backend.Services;

namespace SymptomSense.Backend.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class MachineLearningController : ControllerBase
    {
        private readonly MachineLearningService _mlService;

        public MachineLearningController()
        {
            _mlService = new MachineLearningService();
        }

        [HttpGet]
        public IActionResult Predict()
        {
            var symptoms = new List<string>
            {
                "Fever",
                "Cough"
            };

            string prediction = _mlService.PredictDisease(symptoms);

            return Ok(new
            {
                prediction
            });
        }
    }
}