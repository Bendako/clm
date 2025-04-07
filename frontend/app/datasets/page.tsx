import Link from "next/link";

// Mock datasets data - would be fetched from API
const datasets = [
  {
    id: "dataset-1",
    name: "MNIST Split by Digit",
    description: "MNIST dataset split by digit classes for continual learning",
    type: "image",
    format: "png",
    samples: 60000,
    created_at: "2023-09-10T11:20:00Z",
    updated_at: "2023-09-10T11:20:00Z",
    tags: ["vision", "classification", "digits"],
    tasks: [
      { id: "task-1", name: "Digits 0-1", samples: 12000 },
      { id: "task-2", name: "Digits 2-3", samples: 12000 },
      { id: "task-3", name: "Digits 4-5", samples: 12000 },
      { id: "task-4", name: "Digits 6-7", samples: 12000 },
      { id: "task-5", name: "Digits 8-9", samples: 12000 }
    ]
  },
  {
    id: "dataset-2",
    name: "Customer Support Queries",
    description: "Text dataset of customer support queries for continuous adaptation",
    type: "text",
    format: "json",
    samples: 25000,
    created_at: "2023-10-05T14:30:00Z",
    updated_at: "2023-11-15T09:45:00Z",
    tags: ["nlp", "text", "customer-support"],
    tasks: [
      { id: "task-1", name: "General Inquiries", samples: 5000 },
      { id: "task-2", name: "Technical Support", samples: 8000 },
      { id: "task-3", name: "Billing Issues", samples: 4000 },
      { id: "task-4", name: "Product Questions", samples: 8000 }
    ]
  },
  {
    id: "dataset-3",
    name: "Medical Reports",
    description: "Anonymized medical reports for domain adaptation and transfer learning",
    type: "text",
    format: "txt",
    samples: 18000,
    created_at: "2023-12-01T10:15:00Z",
    updated_at: "2024-01-20T16:45:00Z",
    tags: ["nlp", "text", "medical", "specialized"],
    tasks: [
      { id: "task-1", name: "Diagnosis Reports", samples: 6000 },
      { id: "task-2", name: "Treatment Plans", samples: 4000 },
      { id: "task-3", name: "Patient History", samples: 8000 }
    ]
  },
  {
    id: "dataset-4",
    name: "CIFAR-10 Split",
    description: "CIFAR-10 dataset split by categories for continual learning evaluation",
    type: "image",
    format: "jpg",
    samples: 50000,
    created_at: "2023-08-15T08:30:00Z",
    updated_at: "2023-08-15T08:30:00Z",
    tags: ["vision", "classification", "objects"],
    tasks: [
      { id: "task-1", name: "Animals", samples: 30000 },
      { id: "task-2", name: "Vehicles", samples: 20000 }
    ]
  },
  {
    id: "dataset-5",
    name: "Legal Documents",
    description: "Corpus of legal documents for specialized language understanding",
    type: "text",
    format: "json",
    samples: 12000,
    created_at: "2024-01-08T13:20:00Z",
    updated_at: "2024-02-10T11:30:00Z",
    tags: ["nlp", "text", "legal", "specialized"],
    tasks: [
      { id: "task-1", name: "Contracts", samples: 3000 },
      { id: "task-2", name: "Legal Opinions", samples: 4000 },
      { id: "task-3", name: "Case Law", samples: 5000 }
    ]
  }
];

// Format date helper
const formatDate = (dateString: string) => {
  const date = new Date(dateString);
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  }).format(date);
};

// Format number helper
const formatNumber = (num: number) => {
  return new Intl.NumberFormat('en-US').format(num);
};

// Type badge component
const TypeBadge = ({ type }: { type: string }) => {
  const typeStyles = {
    image: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300",
    text: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
    audio: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300",
    tabular: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300"
  };

  const style = typeStyles[type as keyof typeof typeStyles] || "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300";

  return (
    <span className={`px-2 py-1 text-xs font-medium rounded-full ${style}`}>
      {type}
    </span>
  );
};

export default function DatasetsPage() {
  return (
    <div className="flex flex-col gap-8">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 dark:text-white">Datasets</h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Manage datasets for continual learning tasks
          </p>
        </div>
        <Link
          href="/datasets/upload"
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          aria-label="Upload new dataset"
          tabIndex={0}
        >
          <svg className="h-5 w-5 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          Upload Dataset
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
                placeholder="Search datasets..."
              />
            </div>
          </div>
          <div className="w-full sm:w-48">
            <label htmlFor="type" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Type</label>
            <select
              id="type"
              name="type"
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-gray-700 dark:bg-gray-900 dark:text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            >
              <option value="">All types</option>
              <option value="image">Image</option>
              <option value="text">Text</option>
              <option value="audio">Audio</option>
              <option value="tabular">Tabular</option>
            </select>
          </div>
          <div className="w-full sm:w-48">
            <label htmlFor="sort" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Sort By</label>
            <select
              id="sort"
              name="sort"
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-gray-700 dark:bg-gray-900 dark:text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            >
              <option value="newest">Newest</option>
              <option value="oldest">Oldest</option>
              <option value="name">Name</option>
              <option value="samples">Sample Count</option>
            </select>
          </div>
        </div>
      </div>

      {/* Datasets Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {datasets.map((dataset) => (
          <div key={dataset.id} className="bg-white dark:bg-gray-800 overflow-hidden shadow rounded-lg flex flex-col">
            <div className="px-4 py-5 sm:p-6 flex-1">
              <div className="flex items-center justify-between">
                <h3 className="text-lg leading-6 font-medium text-gray-900 dark:text-white">
                  {dataset.name}
                </h3>
                <TypeBadge type={dataset.type} />
              </div>
              <p className="mt-2 max-w-2xl text-sm text-gray-500 dark:text-gray-400 line-clamp-2">
                {dataset.description}
              </p>
              <div className="mt-4">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-500 dark:text-gray-400">Format</div>
                  <div className="text-sm font-medium text-gray-900 dark:text-white">{dataset.format.toUpperCase()}</div>
                </div>
                <div className="mt-1 flex items-center justify-between">
                  <div className="text-sm text-gray-500 dark:text-gray-400">Samples</div>
                  <div className="text-sm font-medium text-gray-900 dark:text-white">{formatNumber(dataset.samples)}</div>
                </div>
                <div className="mt-1 flex items-center justify-between">
                  <div className="text-sm text-gray-500 dark:text-gray-400">Tasks</div>
                  <div className="text-sm font-medium text-gray-900 dark:text-white">{dataset.tasks.length}</div>
                </div>
                <div className="mt-1 flex items-center justify-between">
                  <div className="text-sm text-gray-500 dark:text-gray-400">Last Updated</div>
                  <div className="text-sm font-medium text-gray-900 dark:text-white">{formatDate(dataset.updated_at)}</div>
                </div>
              </div>
              <div className="mt-4">
                <div className="flex flex-wrap gap-1">
                  {dataset.tags.map((tag, index) => (
                    <span key={index} className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded-full text-xs text-gray-800 dark:text-gray-200">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 px-4 py-4 border-t border-gray-200 dark:border-gray-600 flex justify-between">
              <Link
                href={`/datasets/${dataset.id}`}
                className="text-sm font-medium text-indigo-600 dark:text-indigo-400 hover:text-indigo-500"
                aria-label={`View ${dataset.name} details`}
                tabIndex={0}
              >
                View details
              </Link>
              <Link
                href={`/training/new?dataset=${dataset.id}`}
                className="text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-500"
                aria-label={`Use ${dataset.name} for training`}
                tabIndex={0}
              >
                Train with this dataset
              </Link>
            </div>
          </div>
        ))}
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between">
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
              Showing <span className="font-medium">1</span> to <span className="font-medium">5</span> of <span className="font-medium">5</span> datasets
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