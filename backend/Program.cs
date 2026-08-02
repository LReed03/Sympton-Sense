using Python.Runtime;
using SymptomSense.Backend.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddSingleton<MachineLearningService>();

var app = builder.Build();

PythonEngine.Initialize();

app.Lifetime.ApplicationStopping.Register(() =>
{
    PythonEngine.Shutdown();
});


app.MapControllers();

app.Run();