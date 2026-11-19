using InsuranceRegression.Models;
using Microsoft.ML;

namespace InsuranceRegression.Data;

// Завантаження CSV та розбиття на навчальну/тестову вибірки
public class InsuranceDataLoader
{
    private readonly MLContext _context;

    public InsuranceDataLoader(MLContext context)
    {
        _context = context;
    }

    public DataOperationsCatalog.TrainTestData LoadAndSplit(string path)
    {
        IDataView dataView = _context.Data.LoadFromTextFile<InsuranceData>(
            path,
            hasHeader: true,
            separatorChar: ',');

        return _context.Data.TrainTestSplit(dataView, testFraction: 0.2);
    }
}
