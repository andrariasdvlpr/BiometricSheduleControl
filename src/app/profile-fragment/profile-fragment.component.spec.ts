import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ProfileFragmentComponent } from './profile-fragment.component';

describe('ProfileFragmentComponent', () => {
  let component: ProfileFragmentComponent;
  let fixture: ComponentFixture<ProfileFragmentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ProfileFragmentComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ProfileFragmentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
