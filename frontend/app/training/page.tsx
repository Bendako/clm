import Link from "next/link";

// Mock training jobs data
const trainingJobs = [
  {
    id: "job-1",
    name: "BERT-CLM Medical Domain Training",
    model_id: "model-1",
    model_name: "BERT-CLM-Base",
    dataset_id: "dataset-3",
    dataset_name: "Medical Reports",
    continual_strategy: "EWC",
    status: "completed",
    progress: 100,
    created_at: "2023-12-18T09:30:00Z",
    updated_at: "2023-12-19T16:45:00Z",
    completed_at: "2023-12-19T16:45:00Z",
    tasks_total: 3,
    tasks_completed: 3,
    metrics: {
      accuracy: 0.88,
      forgetting: 0.05,
      training_time: 18500, // in seconds
    }
  },
  {
    id: "job-2",
    name: "ResNet-CLM CIFAR-10 Training",
    model_id: "model-2",
    model_name: "ResNet-CLM",
    dataset_id: "dataset-4",
    dataset_name: "CIFAR-10 Split",
    continual_strategy: "PackNet",
    status: "running",
    progress: 65,
    created_at: "2024-02-20T13:15:00Z",
    updated_at: "2024-02-20T15:30:00Z",
    completed_at: null,
    tasks_total: 2,
    tasks_completed: 1,
    metrics: {
      accuracy: 0.82,
      forgetting: 0.03,
      training_time: 7200, // in seconds so far
    }
  },
  {
    id: "job-3",
    name: "GPT-CLM Legal Domain Training",
    model_id: "model-3",
    model_name: "GPT-CLM-Small",
    dataset_id: "dataset-5",
    dataset_name: "Legal Documents",
    continual_strategy: "Generative Replay",
    status: "queued",
    progress: 0,
    created_at: "2024-02-21T08:45:00Z",
    updated_at: "2024-02-21T08:45:00Z",
    completed_at: null,
    tasks_total: 3,
    tasks_completed: 0,
    metrics: null
  },
  {
    id: "job-4",
    name: "LSTM-CLM Customer Support Training",
    model_id: "model-4",
    model_name: "LSTM-CLM",
    dataset_id: "dataset-2",
    dataset_name: "Customer Support Queries",
    continual_strategy: "LwF",
    status: "completed",
    progress: 100,
    created_at: "2024-01-03T11:20:00Z",
    updated_at: "2024-01-05T10:20:00Z",
    completed_at: "2024-01-05T10:20:00Z",
    tasks_total: 4,
    tasks_completed: 4,
    metrics: {
      accuracy: 0.91,
      forgetting: 0.02,
      training_time: 22800, // in seconds
    }
  },
  {
    id: "job-5",
    name: "ViT-CLM MNIST Training",
    model_id: "model-5",
    model_name: "ViT-CLM",
    dataset_id: "dataset-1",
    dataset_name: "MNIST Split by Digit",
    continual_strategy: "ER+",
    status: "failed",
    progress: 40,
    created_at: "2024-02-14T15:30:00Z",
    updated_at: "2024-02-15T02:45:00Z",
    completed_at: "2024-02-15T02:45:00Z",
    tasks_total: 5,
    tasks_completed: 2,
    metrics: null,
    error: "Out of memory error at task 3"
  }
];

// Format date helper
const formatDate = (dateString: string | null) => {
  if (!dateString) return "N/A";
  
  const date = new Date(dateString);
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  }).format(date);
};

// Format duration helper
const formatDuration = (seconds: number | null | undefined) => {
  if (seconds === null || seconds === undefined) return "N/A";
  
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  
  return hours > 0 
    ? `${hours} hr ${minutes} min` 
    : `${minutes} min`;
};

// Status badge component
const StatusBadge = ({ status }: { status: string }) => {
  const statusStyles = {
    completed: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300",
    running: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
    queued: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300",
    failed: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
  };

  const style = statusStyles[status as keyof typeof statusStyles] || "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300";

  return (
    <span className={`px-2 py-1 text-xs font-medium rounded-full ${style}`}>
      {status}
    </span>
  );
};

// Progress bar component
const ProgressBar = ({ progress }: { progress: number }) => {
  const colorClass = progress === 100 
    ? "bg-green-500" 
    : progress >= 80 
      ? "bg-blue-500" 
      : progress >= 40 
        ? "bg-yellow-500" 
        : "bg-red-500";

  return (
    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
      <div 
        className={`h-2.5 rounded-full ${colorClass}`} 
        style={{ width: `${progress}%` }}
      ></div>
    </div>
  );
};

export default function TrainingPage() {
  return (
    <div className="flex flex-col gap-8">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 dark:text-white">Training Jobs</h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Manage and monitor continual learning training jobs
          </p>
        </div>
        <Link
          href="/training/new"
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          aria-label="Start new training job"
          tabIndex={0}
        >
          <svg className="h-5 w-5 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
          New Training Job
        </Link>
      </div>

      {/* Filters */}
      <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-4">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="w-full sm:w-64">
            <label htmlFor="search" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Search</label>
            <div className="mt-1">
              <input
                type="text"
                name="search"
                id="search"
                className="shadow-sm focus:ring-indigo-500 focus:border-indigo-500 block w-full sm:text-sm border-gray-300 dark:border-gray-700 dark:bg-gray-900 dark:text-white rounded-md"
                placeholder="Search training jobs..."
              />
            </div>
          </div>
          <div className="w-full sm:w-48">
            <label htmlFor="status" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Status</label>
            <select
              id="status"
              name="status"
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-gray-700 dark:bg-gray-900 dark:text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            >
              <option value="">All statuses</option>
              <option value="running">Running</option>
              <option value="queued">Queued</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
            </select>
          </div>
          <div className="w-full sm:w-48">
            <label htmlFor="strategy" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Strategy</label>
            <select
              id="strategy"
              name="strategy"
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-gray-700 dark:bg-gray-900 dark:text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            >
              <option value="">All strategies</option>
              <option value="EWC">EWC</option>
              <option value="LwF">LwF</option>
              <option value="GEM">GEM</option>
              <option value="PackNet">PackNet</option>
              <option value="ER+">ER+</option>
              <option value="Generative Replay">Generative Replay</option>
            </select>
          </div>
        </div>
      </div>

      {/* Training Jobs Table */}
      <div className="bg-white dark:bg-gray-800 shadow overflow-hidden sm:rounded-lg">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Job
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Model & Dataset
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Strategy
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Status
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Progress
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Time
                </th>
                <th scope="col" className="relative px-6 py-3">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {trainingJobs.map((job) => (
                <tr key={job.id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                  <td className="px-6 py-4">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">{job.name}</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">{job.id}</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">Created: {formatDate(job.created_at)}</div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm text-gray-900 dark:text-white">
                      <Link href={`/models/${job.model_id}`} className="text-indigo-600 dark:text-indigo-400 hover:underline">
                        {job.model_name}
                      </Link>
                    </div>
                    <div className="text-sm text-gray-900 dark:text-white mt-1">
                      <Link href={`/datasets/${job.dataset_id}`} className="text-indigo-600 dark:text-indigo-400 hover:underline">
                        {job.dataset_name}
                      </Link>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {job.continual_strategy}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <StatusBadge status={job.status} />
                    {job.error && (
                      <div className="text-xs text-red-600 dark:text-red-400 mt-1">
                        {job.error}
                      </div>
                    )}
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center">
                      <ProgressBar progress={job.progress} />
                      <span className="ml-2 text-sm text-gray-500 dark:text-gray-400">
                        {job.progress}%
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {job.tasks_completed} of {job.tasks_total} tasks
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {job.status === 'completed' ? (
                      <div>
                        <div>Completed: {formatDate(job.completed_at)}</div>
                        <div>Duration: {formatDuration(job.metrics?.training_time)}</div>
                      </div>
                    ) : job.status === 'running' ? (
                      <div>
                        <div>Started: {formatDate(job.created_at)}</div>
                        <div>Running: {formatDuration(job.metrics?.training_time)}</div>
                      </div>
                    ) : (
                      <div>
                        <div>Queued: {formatDate(job.created_at)}</div>
                      </div>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <div className="flex space-x-4 justify-end">
                      <Link
                        href={`/training/${job.id}`}
                        className="text-indigo-600 hover:text-indigo-900 dark:text-indigo-400 dark:hover:text-indigo-300"
                        aria-label={`View details for ${job.name}`}
                        tabIndex={0}
                      >
                        View
                      </Link>
                      {job.status === 'running' && (
                        <button
                          className="text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300"
                          aria-label={`Stop ${job.name}`}
                        >
                          Stop
                        </button>
                      )}
                      {(job.status === 'failed' || job.status === 'queued') && (
                        <button
                          className="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300"
                          aria-label={`Restart ${job.name}`}
                        >
                          Restart
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Pagination */}
      <div className="bg-white dark:bg-gray-800 px-4 py-3 flex items-center justify-between border-t border-gray-200 dark:border-gray-700 sm:px-6 rounded-lg shadow">
        <div className="flex-1 flex justify-between sm:hidden">
          <a href="#" className="relative inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-sm font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700">
            Previous
          </a>
          <a href="#" className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-sm font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700">
            Next
          </a>
        </div>
        <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
          <div>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              Showing <span className="font-medium">1</span> to <span className="font-medium">5</span> of <span className="font-medium">5</span> results
            </p>
          </div>
          <div>
            <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
              <a
                href="#"
                className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm font-medium text-gray-500 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <span className="sr-only">Previous</span>
                <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </a>
              <a
                href="#"
                aria-current="page"
                className="z-10 bg-indigo-50 dark:bg-indigo-900 border-indigo-500 dark:border-indigo-500 text-indigo-600 dark:text-indigo-200 relative inline-flex items-center px-4 py-2 border text-sm font-medium"
              >
                1
              </a>
              <a
                href="#"
                className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm font-medium text-gray-500 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <span className="sr-only">Next</span>
                <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                </svg>
              </a>
            </nav>
          </div>
        </div>
      </div>
    </div>
  );
} 