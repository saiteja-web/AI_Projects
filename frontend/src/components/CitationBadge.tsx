interface CitationBadgeProps {
    page: number;
    section: string;
}

export function CitationBadge({ page, section }: CitationBadgeProps) {
    return (
        <span
            style={{
                display: 'inline-block',
                fontSize: '11px',
                padding: '2px 6px',
                marginLeft: '4px',
                backgroundColor: '#e0e7ff',
                color: '#4338ca',
                borderRadius: '4px',
                border: '1px solid #c7d2fe',
            }}
            title={`Source: ${section}, page ${page}`}
        >
            p.{page} · {section}
        </span>
    );
}
