using Microsoft.AspNetCore.Mvc;

namespace SymptomSense.Backend.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class HealthController : ControllerBase
    {
        [HttpGet]
        public IActionResult Check()
        {
            return Ok(new
            {
                status = "Healthy",
                message = "Backend is running and healthy."
            });
        }
    }
}