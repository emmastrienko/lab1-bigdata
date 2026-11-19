using Microsoft.ML;
using Microsoft.ML.Data;

namespace InsuranceRegression.Evaluation;

// Результат навчання однієї моделі
public sealed record TrainerResult(string Name, ITransformer Model, RegressionMetrics Metrics)
{
    public double RSquared => Metrics.RSquared;
}
