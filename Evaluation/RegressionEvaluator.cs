using InsuranceRegression.Training;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace InsuranceRegression.Evaluation;

// Навчання та оцінювання регресійних моделей, вибір найкращої
public class RegressionEvaluator
{
    private readonly MLContext _context;

    public RegressionEvaluator(MLContext context)
    {
        _context = context;
    }

    public TrainerResult TrainAndEvaluate(string name, IEstimator<ITransformer> pipeline, IDataView trainData, IDataView testData)
    {
        ITransformer model = pipeline.Fit(trainData);
        IDataView predictions = model.Transform(testData);
        RegressionMetrics metrics = _context.Regression.Evaluate(predictions, labelColumnName: "Label");

        return new TrainerResult(name, model, metrics);
    }

    // Вибір найкращої моделі за R^2
    public TrainerResult SelectBest(IEnumerable<TrainerResult> results) =>
        results.OrderByDescending(r => r.RSquared).First();
}
