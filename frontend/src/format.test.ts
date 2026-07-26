import { describe, expect, it } from 'vitest'
import { formatLabel, percent } from './format'

describe('format helpers', () => {
  it('formats machine labels for operators', () => {
    expect(formatLabel('cash_withdrawal')).toBe('Cash Withdrawal')
  })

  it('rounds confidence values', () => {
    expect(percent(0.876)).toBe('88%')
  })
})
