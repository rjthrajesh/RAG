"use client";

interface SourceChipProps {
  pageNumber: number;
  isActive: boolean;
  onClick: () => void;
}

export function SourceChip({ pageNumber, isActive, onClick }: SourceChipProps) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center px-1.5 py-0.5 rounded text-xs font-mono font-medium transition-colors mx-0.5 ${
        isActive
          ? "bg-indigo-600 text-white"
          : "bg-indigo-100 text-indigo-700 hover:bg-indigo-200"
      }`}
    >
      [p.{pageNumber}]
    </button>
  );
}
